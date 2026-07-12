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

    def add(self, **row) -> None:
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("step")

    @property
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
    revive_every: int = 50,
    check_every: int | None = None,
    quiet: bool = False,
) -> History:
    """Train one VQ-VAE. Returns everything that happened.

    loaders: {"learn": ..., "tune": ...} — the grids for this entry (TS or CS).
    steps:   how many batches to learn from. Defaults to settings["steps"].
    """
    steps = steps or settings["steps"]
    check_every = check_every or max(1, steps // 10)
    where = pick_device(settings)
    model.to(where).train()

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.01
    )
    feed = cycle(loaders["learn"])
    history = History()
    started = time.time()

    progress = _Progress(steps, model.codebook.words, len(loaders["learn"].dataset),
                         describe_device(where), enabled=not quiet)

    revived = 0
    running, seen = 0.0, 0
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
            running, seen = 0.0, 0
            progress.checkpoint(step, scored, model.codebook.words)

    history.seconds = time.time() - started
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
def score_predictions(world, loader, where: torch.device, limit: int = 30) -> dict:
    """How often does the GPT name the next day's word correctly?

    Scored against two floors, because "58% correct" means nothing on its own:

      chance      1 / vocabulary. What you would get by guessing at random.
      persistence just say TOMORROW LOOKS LIKE TODAY. This is the honest floor, and it
                  is a HIGH one -- market regimes are sticky, so yesterday's word is
                  usually still right today. A predictor that cannot beat persistence
                  has learned nothing worth having, however good its accuracy looks.
    """
    world.to(where).eval()
    right = sticky = seen = 0
    perplexity, drawing, shrugging, batches = 0.0, 0.0, 0.0, 0

    for i, batch in enumerate(loader):
        if i >= limit:
            break
        out = world(_to(batch, where))
        tokens = out["tokens"]                       # [B, T]
        said = out["said"].argmax(-1)                # [B, T-1]
        answer = tokens[:, 1:]

        right += int((said == answer).sum())
        sticky += int((tokens[:, :-1] == answer).sum())     # "same as yesterday"
        seen += answer.numel()
        perplexity += float(out["perplexity"])
        drawing += float(out.get("drawing_loss", float("nan")))
        shrugging += float(out.get("shrugging", float("nan")))
        batches += 1

    world.train()
    if not seen:
        return {"accuracy": float("nan"), "persistence": float("nan"),
                "chance": float("nan"), "perplexity": 0.0,
                "candle": float("nan"), "shrugging": float("nan")}
    return {
        "accuracy": right / seen,
        "persistence": sticky / seen,
        "chance": 1 / world.words,
        "perplexity": perplexity / max(batches, 1),
        "candle": drawing / max(batches, 1),        # how well it DRAWS tomorrow
        "shrugging": shrugging / max(batches, 1),   # ...vs drawing the average candle
    }


def train_world(world, loaders: dict, settings: dict, steps: int | None = None,
                revive_every: int = 50, check_every: int | None = None,
                quiet: bool = False) -> History:
    """Train the fusion and the predictor together.

    The tokens are still moving while the predictor learns to name them — the codebook
    is being reshaped by this very loss. That is the point (the tokens arrange themselves
    to be worth predicting), but it means the predictor is chasing a target that is
    itself still settling. The codebook moves by slow moving averages, which is what
    keeps that from thrashing.
    """
    steps = steps or settings["steps"]
    check_every = check_every or max(1, steps // 10)
    where = pick_device(settings)
    world.to(where).train()

    learnable = [p for p in world.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(learnable, lr=settings["learning_rate"],
                                  weight_decay=0.01)
    feed = cycle(loaders["learn"])
    history = History()
    started = time.time()

    if not quiet:
        print(f"Training on {describe_device(where)} for {steps:,} steps "
              f"({len(loaders['learn'].dataset):,} sentences)")
        print(f"\n{'step':>6}  {'names it':>9}  {'persistence':>12}  "
              f"{'draws it':>9}  {'shrugging':>10}  {'perplexity':>11}")
        print("  " + "─" * 68)

    revived = 0
    for step in range(1, steps + 1):
        batch = _to(next(feed), where)
        out = world(batch)

        optimiser.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(learnable, 1.0)
        optimiser.step()

        if step % revive_every == 0:
            revived += world.tokenizer.codebook.revive_dead_words(
                out["fused"].detach().reshape(-1, world.tokenizer.width)
            )

        if step % check_every == 0 or step == steps:
            scored = score_predictions(world, loaders["tune"], where)
            history.add(step=step, revived=revived, **scored)
            if not quiet:
                print(f"{step:>6}  {scored['accuracy']:>8.1%}  "
                      f"{scored['persistence']:>11.1%}  "
                      f"{scored['candle']:>9.3f}  {scored['shrugging']:>10.3f}  "
                      f"{scored['perplexity']:>11.1f}")

    history.seconds = time.time() - started
    if not quiet:
        print(f"\n  {history.seconds:.0f}s, {revived:,} dead words revived")
    return history


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
