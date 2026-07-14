"""Teaching a VQ-VAE to compress the market.

The model is given a grid, squeezes it to one word, and tries to rebuild the grid
from that word alone. Training makes the rebuild as accurate as it can be — and
that pressure is what forces the words to come to mean something.

**The number to watch is not the loss. It is the perplexity.**

Perplexity says how many words are actually in use. A VQ-VAE has a nasty habit: a
few words win everything early on, the rest are never chosen again, and the model
settles for describing the entire market with a handful of words. The loss can
look respectable while this happens. Perplexity is what exposes it:

    perplexity ≈ 1        catastrophe — one word for everything, nothing learned
    perplexity ≈ 50       poor — the vocabulary collapsed to a fraction of itself
    perplexity ≈ 300+     healthy — the model is genuinely using its dictionary

We fight the collapse by reviving dead words: any word nobody is choosing gets
dropped onto a real encoder output, so it has somewhere realistic to compete from.

`train()` teaches ONE VQ-VAE (TS or CS) on its own, against a rebuild-the-grid target.
`train_joint()`, further down, teaches the WHOLE model at once -- both VQ-VAEs, the
cross-attention between them, and the GPT that reads their words -- against tomorrow's
candle. There are now TWO dictionaries to watch instead of one, and `train_joint`'s own
docstring explains why its cold start (perplexity really does start at 1.0 there too)
is deliberate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import cycle

import numpy as np
import pandas as pd
import torch

from bubble_bi.settings import device as detect_device


def pick_device(settings: dict) -> torch.device:
    wanted = settings.get("device", "auto")
    if wanted != "auto":
        return torch.device(wanted)
    found = detect_device()
    return torch.device("cuda" if found == "gpu" else "cpu")


def describe_device(where: torch.device) -> str:
    """Name the hardware, so a slow run cannot leave you guessing which one it used."""
    if where.type == "cuda":
        try:
            return f"GPU ({torch.cuda.get_device_name(where.index or 0)})"
        except Exception:
            return "GPU"
    if where.type == "cpu":
        built = torch.version.cuda
        return "CPU" + ("  ⚠️ torch is a CPU-only build" if built is None else "")
    return str(where).upper()


def _to(batch: dict, where: torch.device) -> dict:
    return {k: v.to(where) if torch.is_tensor(v) else v for k, v in batch.items()}


@dataclass
class History:
    """What happened during training. Every number here was measured, not guessed."""

    rows: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    best_step: int = 0

    def add(self, **row) -> None:
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("step")

    def last(self) -> dict:
        return self.rows[-1] if self.rows else {}


@torch.no_grad()
def evaluate(model, loader, where: torch.device, limit: int = 40) -> dict:
    """How well does the token rebuild a window it has never seen?

    Against THREE floors, because a single one lets you fool yourself.

      long-run average   Predict zero everywhere. The features are normalised, so zero IS
                         the long-run average. ⚠️ A WEAK BAR: a token that knew nothing
                         but "this window sits above its usual level" would already beat
                         it. This is the bar we used to report against, and it flattered
                         us badly.

      the window's own   Predict the average of THIS window — one number per feature.
      average            ⚠️ THIS IS THE BAR THAT MATTERS. It hands the model the level for
                         free and asks whether it knows anything about the SHAPE inside
                         the window. Measured: our token LOSES to it.

      repeat the last    Predict every day as a copy of the final day. Intuitive — but
      day                measured, it is WORSE than predicting zero, because one day is a
                         noisy sample. So it FLATTERS the model. It is reported only so
                         that nobody is tempted to headline it.
    """
    model.to(where).eval()
    token = flat = level = last = 0.0
    batches, chosen = 0, []

    for i, batch in enumerate(loader):
        if i >= limit:
            break
        batch = _to(batch, where)
        out = model(batch)
        token += float(out["rebuild_loss"])

        grid = batch["grid"]                                     # [B, C, days, F]
        present = batch.get("present")
        weight = (present.unsqueeze(-1).unsqueeze(-1).to(grid.dtype).expand_as(grid)
                  if present is not None else torch.ones_like(grid))
        total = weight.sum().clamp(min=1)

        def cost(guess):
            return float((((grid - guess) ** 2) * weight).sum() / total)

        flat += cost(torch.zeros_like(grid))              # the long-run average
        level += cost(grid.mean(dim=2, keepdim=True))     # THIS window's own average
        last += cost(grid[:, :, -1:, :])                  # the final day, repeated

        chosen.append(out["ids"].cpu())
        batches += 1

    model.train()
    if not batches:
        return {"rebuild": float("nan"), "guessing": float("nan"),
                "window_mean": float("nan"), "last_day": float("nan"),
                "perplexity": 0.0, "words_used": 0}

    ids = torch.cat(chosen)
    counts = torch.bincount(ids, minlength=model.codebook.words).float()
    p = counts / counts.sum()
    live = p[p > 0]
    return {
        "rebuild": token / batches,
        "guessing": flat / batches,          # the weak bar
        "window_mean": level / batches,      # THE bar
        "last_day": last / batches,          # flattering; here so nobody headlines it
        "perplexity": float(torch.exp(-(live * live.log()).sum())),
        "words_used": int((counts > 0).sum()),
    }


def train(
    model,
    loaders: dict,
    settings: dict,
    steps: int | None = None,
    entry: str | None = None,
    revive_every: int = 50,
    check_every: int | None = None,
    patience: int = 5,
    quiet: bool = False,
    on_check=None,
) -> History:
    """Train one VQ-VAE, and STOP when it starts getting worse.

    loaders: {"learn": ..., "tune": ...} — the grids for this entry (TS or CS).
    steps:   the MOST batches to learn from. Defaults to settings["steps"].
    patience: give up after this many checks with no improvement on the held-out days.
    on_check: called as on_check(step, scored) at every held-out check. A hyperparameter
              search uses this to give up on a hopeless trial early. It may RAISE to stop
              training — the exception is deliberately not caught.

    ⚠️ Why this exists. CS has ~2,600 grids; TS has ~78,000. They share one `steps`
    setting, so 10,000 steps is 33 passes over the TS data and **243 passes** over the
    CS data. On a real run CS's held-out error bottomed out at step 1,000 and then climbed
    steadily for the next nine thousand — 0.90 → 1.03, barely better than guessing —
    while its codebook decayed from 187 words back down to 141. Every one of those steps
    made the model worse, and without this it would have kept the wreckage.

    So: we keep the BEST model we ever saw, not the last one.

    entry: "ts" or "cs" — takes that entry's own step budget if it has one, because CS
           needs far fewer than TS does.
    """
    if steps is None and entry:
        steps = settings.get(entry, {}).get("steps")
    steps = steps or settings["steps"]
    check_every = check_every or max(1, steps // 10)
    where = pick_device(settings)
    model.to(where).train()

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    feed = cycle(loaders["learn"])
    history = History()
    started = time.time()

    progress = _Progress(steps, model.codebook.words, len(loaders["learn"].dataset),
                         describe_device(where), enabled=not quiet)

    revived = 0
    running, seen = 0.0, 0
    best, best_at, best_weights, stale = float("inf"), 0, None, 0

    for step in range(1, steps + 1):
        batch = _to(next(feed), where)
        out = model(batch)

        optimiser.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        # Words nobody is choosing are restarted on real data, so they have somewhere
        # realistic to compete from. Without this the dictionary quietly collapses.
        if step % revive_every == 0:
            revived += model.codebook.revive_dead_words(out["summary"].detach())

        here = float(out["rebuild_loss"].detach())
        running += here
        seen += 1
        progress.tick(step, here, float(out["perplexity"].detach()))

        if step % check_every == 0 or step == steps:
            scored = evaluate(model, loaders["tune"], where)
            history.add(
                step=step,
                # what it scores on the days it is LEARNING from...
                learning=running / max(seen, 1),
                # ...against the days it has never seen. If these two part company, it
                # is memorising rather than learning.
                rebuild=scored["rebuild"],
                guessing=scored["guessing"],
                window_mean=scored["window_mean"],
                last_day=scored["last_day"],
                perplexity=scored["perplexity"],
                words_used=scored["words_used"],
                revived=revived,
            )
            if on_check is not None:
                on_check(step, scored)      # may raise, on purpose: that is how a search prunes
            running, seen = 0.0, 0
            progress.checkpoint(step, scored, model.codebook.words)

            # Keep the best model we ever saw, on days it was not trained on.
            if scored["rebuild"] < best - 1e-4:
                best, best_at, stale = scored["rebuild"], step, 0
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    if not quiet:
                        print(f"\n  ⏹  Stopping at step {step:,}: the held-out error has "
                              f"not improved for {patience} checks.")
                    break

    history.seconds = time.time() - started
    history.best_step = best_at

    # Hand back the BEST model, not the last one. The last one may be a wreck.
    if best_weights is not None and best_at != history.rows[-1]["step"]:
        model.load_state_dict(best_weights)
        if not quiet:
            print(f"  ↩︎  Kept the model from step {best_at:,} (held-out {best:.3f}) — "
                  f"everything after it was worse.")

    progress.done(history.seconds, revived)
    return history


class _Progress:
    """A live bar, so a long training run does not look like a hung notebook."""

    REDRAW_EVERY = 0.15          # seconds — smooth to the eye, cheap on the notebook

    def __init__(self, steps: int, words: int, grids: int, where: str, enabled: bool):
        self.steps, self.words, self.enabled = steps, words, enabled
        self.started = time.time()
        self.last_drawn = 0.0
        self.line = ""
        if enabled:
            print(f"Training on {where} for {steps:,} steps "
                  f"({grids:,} grids to learn from)")
            print(f"\n{'step':>7}  {'rebuild':>8}  {'vs guessing':>12}  "
                  f"{'perplexity':>11}  {'words used':>12}")
            print("  " + "─" * 60)

    def tick(self, step: int, loss: float, perplexity: float) -> None:
        if not self.enabled:
            return
        now = time.time()
        # Throttled by the clock, not the step count: a fast GPU run and a slow CPU
        # one then both redraw at a readable pace, and neither floods the notebook.
        if now - self.last_drawn < self.REDRAW_EVERY and step != self.steps:
            return
        self.last_drawn = now

        done = step / self.steps
        filled = int(done * 28)
        gone = now - self.started
        left = gone / max(done, 1e-9) - gone
        bar = "█" * filled + "░" * (28 - filled)
        self._draw(
            f"  {bar} {done:>4.0%}  step {step:,}/{self.steps:,}  "
            f"rebuild {loss:.3f}  perplexity {perplexity:>5.1f}  "
            f"~{left:>3.0f}s left"
        )

    def checkpoint(self, step: int, scored: dict, words: int) -> None:
        if not self.enabled:
            return
        self._draw("")                       # wipe the bar, print the real row under it
        share = scored["rebuild"] / max(scored["guessing"], 1e-9)
        print(f"{step:>7}  {scored['rebuild']:>8.3f}  {share:>11.0%}  "
              f"{scored['perplexity']:>11.1f}  {scored['words_used']:>6} / {words}")

    def done(self, seconds: float, revived: int) -> None:
        if not self.enabled:
            return
        self._draw("")
        print(f"\n  {seconds:.0f}s, {revived:,} dead words revived along the way")

    def _draw(self, text: str) -> None:
        import sys

        pad = " " * max(0, len(self.line) - len(text))
        sys.stdout.write("\r" + text + pad + ("\r" if not text else ""))
        sys.stdout.flush()
        self.line = text


@torch.no_grad()
def score_joint(world, loader, where: torch.device, limit: int = 30) -> dict:
    """How the joint model is doing, against its two HONEST floors.

        drawing  vs  shrugging     -- draw tomorrow's candle, vs draw the average candle
        accuracy vs  persistence   -- name tomorrow's words, vs say today's words again

    Never against zero. This project has had to walk back two numbers that were quoted
    without their floor.
    """
    world.to(where).eval()
    total = {k: 0.0 for k in ("drawing", "shrugging", "accuracy", "persistence",
                              "ts_perplexity", "cs_perplexity")}
    seen = 0
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        out = world(_to(batch, where))
        total["drawing"] += float(out["drawing_loss"])
        total["shrugging"] += float(out["shrugging"])
        total["accuracy"] += float(out["accuracy"])
        total["persistence"] += float(out["persistence"])
        total["ts_perplexity"] += float(out["ts_perplexity"])
        total["cs_perplexity"] += float(out["cs_perplexity"])
        seen += 1
    world.train()
    return {k: v / max(seen, 1) for k, v in total.items()}


def train_joint(world, loaders: dict, settings: dict, steps: int | None = None,
                revive_every: int = 50, check_every: int | None = None,
                patience: int = 5, quiet: bool = False) -> History:
    """Everything at once: the encoders, BOTH codebooks, the fusion and the GPT.

    ⚠️ COLD START, on purpose. There is no reconstruction-only warm-up, not even a short
    one: pretraining would shape the representation for the compress-everything objective
    this whole design exists to escape.

    The price is that the GPT spends its first steps predicting a vocabulary of roughly
    one word (perplexity really does start at 1.0). Watch `ts_perplexity` and
    `cs_perplexity` from step one -- they are printed every check, from the very first
    one. If they never open, the cold start has dead-locked -- and the fix is a 300-step
    reconstruction-only warm-up, added WITH EVIDENCE rather than on principle.

    ⚠️ MEASURED, not assumed: a cold start's codebook does NOT climb out of that opening
    collapse on its own. Not via the reconstruction anchor, not via the diversity loss,
    however high it is turned up. Perplexity sat at EXACTLY 1.0 -- one word for
    everything -- at every single check before the first dead-word revival, in every
    run tried. The ONLY thing that opens the dictionary is `revive_dead_words()`:
    dropping the words nobody is choosing onto real encoder output, so they have
    somewhere realistic to compete from.

    That makes `revive_every` load-bearing, not a tuning detail -- and it sets a trap.
    `check_every` defaults to `steps // 10`. On any run shorter than about 500 steps,
    that is SMALLER than `revive_every`'s default of 50 -- so the first time this
    function judges the model (and, with `patience` checks of no improvement, decides
    whether to throw it away) can land BEFORE the codebook has ever had one revival.
    A run would then be scored, and possibly killed, on a dictionary that never had a
    chance to open -- while the loss curve looks perfectly reasonable the whole time.

    So we clamp `revive_every` down to at most `check_every`: at least one revival is
    guaranteed before the first check, always.

    A live bar (`_JointProgress`) ticks on EVERY step, not just at `check_every` -- a
    multi-hour Colab run used to print only the ~10 checkpoint lines below and go
    silent in between, indistinguishable from a hung notebook. `_Progress` (used by
    `train()`) is not reused here: its `tick`/`checkpoint` are shaped for a single
    VQ-VAE's numbers (rebuild/guessing/perplexity/words_used), and this run has an
    entirely different set (candle/shrugging/accuracy/persistence/two perplexities).
    """
    steps = steps or settings["steps"]
    check_every = check_every or max(1, steps // 10)
    # See the docstring above: a revival must land before the first judgement of the
    # model, or the very first check (and possibly the whole run, via early stopping)
    # can be decided on a codebook that has never once been given the chance to open.
    revive_every = min(revive_every, max(1, check_every))
    where = pick_device(settings)
    world.to(where).train()

    optimiser = torch.optim.AdamW(world.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    feed = cycle(loaders["learn"])
    history = History()
    started = time.time()
    best, best_at, best_weights, stale = float("inf"), 0, None, 0
    revived = 0

    progress = _JointProgress(steps, describe_device(where), enabled=not quiet)

    for step in range(1, steps + 1):
        out = world(_to(next(feed), where))
        optimiser.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1.0)
        optimiser.step()

        # Dead words get dropped onto real encoder output, so they have somewhere realistic
        # to compete from. Both dictionaries -- there is no fused codebook to revive now.
        if step % revive_every == 0:
            revived += world.tokenizer.ts.codebook.revive_dead_words(
                out["ts_summary"].detach())
            revived += world.tokenizer.cs.codebook.revive_dead_words(
                out["cs_summary"].detach())

        progress.tick(step, float(out["drawing_loss"].detach()),
                     float(out["ts_perplexity"].detach()), float(out["cs_perplexity"].detach()))

        if step % check_every == 0 or step == steps:
            scored = score_joint(world, loaders["tune"], where)
            history.add(step=step, revived=revived, **scored)
            if not quiet:
                progress.checkpoint()
                print(f"  {step:>6}  candle {scored['drawing']:.3f} "
                      f"(shrug {scored['shrugging']:.3f})   "
                      f"words {scored['accuracy']:.1%} "
                      f"(persist {scored['persistence']:.1%})   "
                      f"perplexity TS {scored['ts_perplexity']:.0f} "
                      f"CS {scored['cs_perplexity']:.0f}")

            # Keep the best model on how well it DRAWS tomorrow -- naming accuracy is the
            # number that flatters a collapsed codebook, so it must never be what we select on.
            if scored["drawing"] < best - 1e-4:
                best, best_at, stale = scored["drawing"], step, 0
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in world.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    if not quiet:
                        print(f"\n  ⏹  Stopped at step {step:,} — tomorrow's candle has not "
                              f"improved for {patience} checks.")
                    break

    history.seconds = time.time() - started
    history.best_step = best_at
    if best_weights is not None:
        world.load_state_dict(best_weights)
    progress.done(history.seconds)
    return history


class _JointProgress:
    """A live bar for `train_joint`, so a multi-hour run does not look like a hung
    notebook. See `_Progress` for the same idea, built for `train()`'s own numbers --
    this one exists because `train_joint`'s checks carry a different set entirely
    (candle/shrugging/accuracy/persistence/two perplexities), not the single
    rebuild/perplexity/words_used triple `_Progress` was built to show.
    """

    REDRAW_EVERY = 0.15          # seconds — smooth to the eye, cheap on the notebook

    def __init__(self, steps: int, where: str, enabled: bool):
        self.steps, self.enabled = steps, enabled
        self.started = time.time()
        self.last_drawn = 0.0
        self.line = ""
        if enabled:
            print(f"Training jointly on {where} for {steps:,} steps")

    def tick(self, step: int, drawing: float, ts_perplexity: float, cs_perplexity: float) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_drawn < self.REDRAW_EVERY and step != self.steps:
            return
        self.last_drawn = now

        done = step / self.steps
        filled = int(done * 28)
        gone = now - self.started
        left = gone / max(done, 1e-9) - gone
        bar = "█" * filled + "░" * (28 - filled)
        self._draw(
            f"  {bar} {done:>4.0%}  step {step:,}/{self.steps:,}  "
            f"candle {drawing:.3f}  perplexity TS {ts_perplexity:>5.1f} "
            f"CS {cs_perplexity:>5.1f}  ~{left:>3.0f}s left"
        )

    def checkpoint(self) -> None:
        """Wipe the bar so the checkpoint row underneath it is not smeared together."""
        if self.enabled:
            self._draw("")

    def done(self, seconds: float) -> None:
        if not self.enabled:
            return
        self._draw("")
        print(f"\n  {seconds:.0f}s")

    def _draw(self, text: str) -> None:
        import sys

        pad = " " * max(0, len(self.line) - len(text))
        sys.stdout.write("\r" + text + pad + ("\r" if not text else ""))
        sys.stdout.flush()
        self.line = text


def baseline_rebuild(loader, limit: int = 40) -> float:
    """What you would score by simply guessing the average for everything.

    Close to 1.0, because the features were normalised to spread 1 — but measured,
    not assumed.
    """
    total, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        grid = batch["grid"]
        present = batch.get("present")
        if present is not None:
            w = present.unsqueeze(-1).unsqueeze(-1).to(grid.dtype)
            total += float((grid.pow(2) * w).sum() / w.expand_as(grid).sum().clamp(min=1))
        else:
            total += float(grid.pow(2).mean())
        seen += 1
    return total / max(seen, 1)


def word_usage(model, loader, where: torch.device | None = None,
               limit: int = 200) -> np.ndarray:
    """How often each word gets chosen — for looking at what the model learned."""
    where = where or next(model.parameters()).device
    model.eval()
    counts = torch.zeros(model.codebook.words)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            ids = model(_to(batch, where))["ids"].cpu()
            counts += torch.bincount(ids, minlength=model.codebook.words).float()
    model.train()
    return counts.numpy()
