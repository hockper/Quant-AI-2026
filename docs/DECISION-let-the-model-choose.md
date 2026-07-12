# Decision: let the model tell us what matters

**Taken deliberately. Do not "fix" this without reading it first.**

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
