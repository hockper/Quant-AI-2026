# Decision: let the model tell us what matters

**Taken deliberately. Do not "fix" this without reading it first.**

---

## ⚠️ UPDATE, 2026-07-13 — THE REGIME PIVOT IS OFF. Read this before the rest.

Everything below was measured on an **untuned, collapsing CS**. The first real
hyperparameter search (`bb.tuning`, on a T4) changed the answer, and it changed it in the
half of the model nobody was watching.

Scored against the noise floor of the measurement itself:

| | rows to probe on | noise floor (sd) | best `direction` | verdict |
|---|---|---|---|---|
| **TS** (one stock) | 78,000 | 0.006 | **0.030** | at the noise floor — **nil** |
| **CS** (whole market) | ~600 | 0.046 | **0.989** | **21 standard errors clear** |

**The candle is noise at the STOCK level and signal at the MARKET level.**

- **TS** sees fifteen days of *one* company's candles: fifteen independent random numbers,
  incompressible. It throws direction away. **Everything below still holds for TS.**
- **CS** sees thirty companies on the *same day*. They all move together — the **market
  factor** is the single highest-variance, most compressible thing in that grid, and one
  number explains a great deal of it. So CS spends its words on it, and **today's market
  direction survives the bottleneck almost perfectly.**

That is textbook finance falling out of a compression argument, and nobody told it:
**idiosyncratic direction is noise; systematic direction is not.**

So the claim below — *"both halves independently kept the regime and discarded the
direction"* — **is false for CS.** The token does carry direction: the market's. The
proposed pivot (retarget the prediction and the RL agent onto volatility/regime) is
therefore **not** justified, and is off.

**What this does NOT say.** This is today's direction *surviving the bottleneck*, not
tomorrow's direction being *predictable*. The tokenizer keeping it is necessary, not
sufficient. Whether it is forecastable is the **predictor's** question, and is measured
when the predictor is tuned — not here.

**A caution about the numbers.** `before_quant` in the same table is not to be trusted and
is now suppressed: with ~600 CS rows and a 256-wide dense probe it had as many free
parameters as data, and it printed an impossible −4.164 against a token, quantised *from
that very vector*, scoring +0.561. Quantising cannot add information. See
`_score_and_floor` in `bubble_bi/tuning.py`.

---

## The original decision, and the evidence it was taken on (now superseded for CS)

## What we found

Given the candle — the gap, the body, the two wicks, all handed to it explicitly — the
tokenizer **threw it away**.

TS, after a proper GPU run (perplexity 381 of 512, 479 words in use, a healthy codebook):

```
volatility        50%
memory            49%
price             45%
microstructure    38%
flow              11%
candle             3%      <- the thing you would actually trade on
```

And per feature: `macd` 85%, `rsi` 75%, `sma_ratio_20` 73% — against `gap` 4%, `body` 5%,
`log_return` 9%, `lower_wick` **−1%** (worse than guessing the mean).

CS said the same thing about the market: its words explain **88%** of how *violent* a day
was and **8%** of which *way* it went — the latter *below* its own shuffled floor.

**Both halves of the tokenizer, independently, kept the regime and discarded the
direction.**

## Why it is right to

One token is **9 bits** (log₂ 512). Fifteen days of MACD is a *smooth curve* — a handful
of numbers describe it. Fifteen days of candle bodies are *fifteen independent random
numbers* — incompressible. A 9-bit code spends itself on what can be compressed. There is
no way for it not to.

This is not the model failing. It is the model reporting, correctly, that daily direction
is close to noise and volatility regime is not.

## The decision

**We do NOT force the token to keep the candle.**

We could. Weight the candle features up in the reconstruction loss, or bolt a head onto
the token that predicts the candle directly, and it would comply. But it would be
complying with us, not with the data — and a predictor built on a token that was *made*
to memorise noise is a predictor built on sand.

So instead: **finish the pipeline as it stands, and if the candle turns out not to matter,
change the PREDICTION and the RL AGENT to work with what the model says does matter.**

## What that pivot would look like

The current downstream task assumes direction is the prize:

| now | would become |
|---|---|
| predict the next **candle** | predict the next **regime** — is tomorrow calm or violent, trending or reverting |
| RL agent goes **long / short** on direction | RL agent sizes **exposure** by regime, and trades **volatility** rather than direction |

That is not a retreat. Volatility is *forecastable* in a way direction is not — it
clusters, it persists, and the tokenizer found that on its own without being told. A
strategy that sizes positions by a regime it can actually see is honest. One that bets on
a direction the model has told us is noise is not.

## What would change this

If, after the full GPU run, the fusion token DOES carry direction — the autopsy's
"TOMORROW's return, from the token" climbs meaningfully above its luck floor — then the
candle is not noise after all and the original plan stands.

Right now that number is **negative** (worse than guessing the average). Watch it.

See also: [`OPEN-QUESTION-codebook-collapse.md`](OPEN-QUESTION-codebook-collapse.md).
