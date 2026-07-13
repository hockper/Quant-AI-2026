# Train it all at once: TS + CS + fusion + predictor

**Date:** 2026-07-13
**Status:** design, approved for planning
**Supersedes:** the two-stage scheme (train tokenizer → freeze → train predictor) and the
single fused codebook.

## Why

The tokenizer is trained to **rebuild the present**. The predictor needs it to **carry the
future**. Those are different jobs, and we have now measured, three separate ways, that the
first does not deliver the second:

1. **The tokenizer throws away what it is not asked for.** Handed the candle explicitly, the
   best compressor discarded it (`docs/DECISION-let-the-model-choose.md`).
2. **Our tuning objective was gameable.** "Does today survive the bottleneck?" is trivially
   maximised by making the window *be* today — because what we ask it to preserve is already
   sitting in its input. The search found the loophole and drove `days` to the floor. TS's
   direction score went 0.03 → 0.22 buying nothing but a smaller window.
3. **A target that is not in the input cannot be gamed that way.** Tomorrow is not in the
   grid. To score on it, the token must carry something genuinely useful, and discarding
   history starts to *cost*.

**So the parts must be trained against the thing we actually want.** One loss, one optimiser,
everything at once.

STORM does exactly this (§5.1.4): the dual VQ-VAE, the factor module and the return
predictor are optimised together, with reconstruction as a weak anchor rather than a
pretraining stage.

## The landmine this design exists to defuse

**Our GPT invents its own vocabulary and is then graded on predicting it.**

```
naming loss = CrossEntropy( GPT(z₁…z_t),  id(z_{t+1}) )
                            └ the model's ┘ └ ALSO the model's ┘
```

Two ways to drive that to zero: learn real dynamics (hard), or **make `id(z)` constant**
(trivial). Gradient descent takes the second. It is a *stable* fixed point — once the
vocabulary is degenerate nothing pulls it back.

Measured, on our own model: **92% next-token accuracy at perplexity 2.2.** In NLP this cannot
happen because the tokenizer is external and fixed. Here it is not.

Today the codebook survives only because reconstruction is trained **first and separately**.
**Joint training removes that protection.** It does not cause the collapse — it exposes it.

Every codebook in this project that had a reconstruction anchor stayed healthy; the one that
did not, collapsed:

| codebook | its anchor | perplexity |
|---|---|---|
| TS | rebuild today's grid | **157** ✅ |
| CS | rebuild today's grid | healthy ✅ |
| **fusion** | *draw tomorrow's candle* | **10** ❌ |

## The architecture

```
                    ONE loss, ONE optimiser, from step zero

  CS grid  ──► CS encoder ──► z_cs ──┬──► CS codebook ──► cs_token
  (30 cos,                            │         │
   1 day)                             │         └──► CS decoder ──► rebuild CS grid  [ANCHOR]
                                      │ (keys / values)
                                      ▼
  TS grid  ──► TS encoder ──► z_ts ──► cross-attention ──► z_ts′
  (1 co,                                                    │
   2 days)                                     TS codebook ◄┘
                                                    │
                                     ts_token ◄─────┤
                                                    └──► TS decoder ──► rebuild TS grid [ANCHOR]

             GPT reads the sentence:  (ts,cs)₁ (ts,cs)₂ … (ts,cs)_t
                        │
             ┌──────────┴──────────┐
        head A: next word          head B: tomorrow's candle
        (target STOP-GRADDED)      (the real objective)
```

### Four decisions, and what each one buys

**1. TWO tokens per day. No fused codebook.**
TS and CS each keep their own codebook **and decoder**, anchored by rebuilding their own
grid — the arrangement we have *measured* to be healthy (TS 323 words alive, CS 129). The
fused codebook, the only one the predictor ever saw, is the one we watched collapse. We
delete it rather than keep patching it.

It is also what makes this affordable: **the CS grid is identical for every company on a
day**, so it is encoded and decoded **once per day**, not once per company-day. A 30×
saving, and the reason a two-token design is cheaper than the one-token design it replaces.

The story survives, and improves: each day is described by two words — *what this stock did*
and *what the market did* — and we have measured that these carry **different things** (TS:
volatility, ~no direction; CS: the market's direction, 0.94).

**2. Cross-attention BEFORE quantisation, inside the TS path.**
`z_ts` reads the market's cells and comes out context-aware; *then* the TS codebook quantises
it. So `ts_token` = "what this stock did, **given** what the market was doing".

It has to happen here. After a 9-bit quantisation the fine detail the attention needs is
already destroyed. Reconstruction keeps that token honest; **prediction is what pays for the
market context** — rebuilding the TS grid does not need CS, so only the forecast can justify
carrying it.

⚠️ We measured the old attention map as **FLAT** — every company read PG most; attention
depended on *which company is read*, never on *who is reading*. It may stay flat. But it was
trained through frozen encoders into a collapsed codebook, so we never learned whether the
*idea* fails or the *training* did. Now we will.

**3. The naming loss keeps its head, but loses its teeth.**
The target token is computed under `torch.no_grad()`. The gradient path from `naming` into
the encoders and the codebook is **physically severed**. It can make the GPT better at
predicting the language; it can never make the language easier.

This is SimSiam's result: **stop-gradient + an asymmetric predictor head is enough** to stop
representation collapse — no negatives, no EMA. **The GPT is already that predictor head.**

And we already hold the other half of DINO: our **diversity loss** (STORM eq. 4 — maximise
the entropy of the averaged soft assignment) *is* DINO's centering, the force that stops one
word dominating. We built the brake and left the accelerator floored. This lifts the foot.

**If perplexity still slides**, the next step is a proper **EMA teacher** (BYOL/DINO): the
target comes from a slowly-updated copy of the tokenizer, so it cannot be chased into
degeneracy. We add it **on evidence, not on principle** — perplexity makes the failure
visible from step 1.

**4. Cold start.**
One loss from step zero. No reconstruction-only pretraining, not even briefly: pretraining
would bake in the compress-everything bias this whole redesign exists to escape. STORM does
the same.

The risk is real and is stated in the Risks section.

## The loss

```
L =  w_predict · MSE(tomorrow's candle)             the real objective
  +  w_naming  · [ CE(ts) + CE(cs) ]                target under no_grad()
  +  w_recon   · [ MSE(TS grid) + MSE(CS grid) ]    the ANCHORS
  +  w_commit  · [ commit_ts + commit_cs ]
  +  w_div     · [ div_ts + div_cs ]
```

### ⚠️ STORM's loss weights are NOT portable. Do not copy them.

STORM reports reconstruction at `1e-3` against prediction at `0.1` — reconstruction looks
100× weaker. **That number is meaningless out of context.** Their reconstruction is
`‖x − x′‖²`, a Frobenius norm — a **sum** over millions of elements. Ours is a `.mean()`. A
`1e-3` weight on a sum of ~4M terms is an *enormous* weight on the mean.

Copy `1e-3` into our code and **we delete the anchor** — precisely the failure this design
exists to prevent.

**Starting point:** our own measured values. `w_recon = 1.0` (mean-reduced; the setting under
which TS reaches perplexity 157), `w_commit = 1.0` (STORM's, and what our own search asked
for — the "literature" 0.25 we adopted was wrong and both the search and the paper say so),
`w_div = 0.1`, `w_naming = 0.1`.

`w_predict` is **searched**. It is the one weight nobody has ever set from evidence, and it
is now the most important number in the model.

## What proves it works

| watch | the honest bar | why it matters |
|---|---|---|
| **perplexity, BOTH codebooks** | stays in the hundreds | **THE number.** If either slides, the anchor or the stop-gradient failed. Watch from step 1, not at the end. |
| next-token accuracy | **persistence** ("tomorrow's word = today's word") | never against zero |
| tomorrow's candle | **shrugging** (draw the average candle) | never against zero |
| the attention map | is it FLAT? | a flat map means the fusion is doing nothing, and no loss curve will ever say so |

### The bar that finally became honest

With `cs_days = 1`, **consecutive CS grids share nothing at all**. So CS-persistence collapses
toward chance and the next-token task is honest *for the first time*.

The old "predictor loses to persistence at 65%" result was substantially an **artefact**: at
`days = 15`, day *t*'s window and day *t+1*'s window overlap by **14 days — 93% of their
content**. Consecutive tokens were near-identical *by construction*. We built a task whose
trivial baseline could not lose, and then blamed the model.

## Risks, stated plainly

1. **Cold start dead-locks.** The GPT spends early steps predicting a ~1-word vocabulary
   (perplexity genuinely starts at 1.0 — we have watched it). If the codebooks never open,
   the GPT learns to predict noise and the two can lock. **Detection:** perplexity from step
   1. **Fallback:** a 300-step reconstruction-only warm-up — added *with evidence*, not
   pre-emptively.
2. **Compute.** Nothing is frozen, so the ~74MB encoder cache goes away and every step
   re-encodes. Mitigated by encoding CS **once per day** rather than once per company-day.
   If it is still too slow, shorten the sentence before weakening the model.
3. **Naming flatlines.** If tomorrow's word is simply not predictable, the naming loss sits
   at chance. **That is an honest null, not a failure** — and it is exactly why `candle`
   carries the objective and `naming` only rides along at 0.1.
4. **The attention stays flat.** Possible. But then we will have learned that the *idea*
   fails, rather than that the training did — which is more than we know today.

## Testing

- **The naming loss cannot touch the vocabulary.** A gradient test: backprop `naming` alone
  and assert the TS/CS encoder and codebook parameters receive **zero** gradient. This is the
  single most important test in the design — it is the severed collapse channel, asserted.
- **The anchors are real.** Backprop the reconstruction term alone and assert it *does* reach
  the encoders and codebooks.
- **The fusion actually reads CS.** Perturb the CS grid and assert `ts_token`'s pre-quantised
  vector changes. If it does not, the cross-attention is decorative.
- **CS is encoded once per day, not once per company-day.** Assert the CS encoder is called
  `n_days` times for a batch spanning `n_days × n_companies` samples. This is the saving the
  whole design rests on; if it silently regresses, training becomes 30× slower and nothing
  says so.
- **Codebooks survive a joint run.** On a small synthetic panel, train end-to-end and assert
  both perplexities finish above a floor. This is the test that would have caught the fusion
  collapse.
- **Both bars are computed and reported** — persistence for the token, shrugging for the
  candle. A metric without its floor gets quoted.

## Out of scope

- **The RL agent.** It consumes the GPT's hidden state and is Spec 2. Unchanged.
- **An EMA teacher.** Added only if perplexity slides under stop-gradient alone.
- **Contrastive / orthogonality losses.** STORM has both. Our codebook is EMA-updated with no
  gradient on the dictionary, so their orthogonality loss (eq. 5) would be **inert** for us.
  Revisit only if diversity proves insufficient.
- **Re-tuning TS and CS in isolation.** Their standalone hyperparameters are now the *wrong*
  question — the joint objective is what matters, and `w_predict` is the knob to search.
