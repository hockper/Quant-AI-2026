# Open question: the fusion codebook collapses

**Status: diagnosed, not fixed. Revisit after the Colab run.**

## What happens

Training the fusion + predictor together, the fusion's codebook collapses to ~10 words
out of 512. It does this on every loss balance we tried.

```
                                      names it   persistence   draws it   shrugging   perplexity
naming 1.0, commit 0.25, div 0.1        69.7%       73.2%       1.152       1.177         7.2
naming 0.1, commit 1.0,  div 0.5        23.4%       66.1%       1.141       1.177         9.8
```

Early in training it is even starker: **perplexity 2.2, "accuracy" 92%.** The model has
found the shortcut — make every day the same word, and the next word becomes trivially
predictable. Reviving dead words breaks the collapse open (perplexity jumps to ~7), but
it never climbs past ~10.

## Why (the diagnosis)

Compare the two codebooks. They are the **same `Codebook` class**, same diversity loss,
same commitment loss:

| | its anchor | perplexity |
|---|---|---|
| **TS** | rebuild **today's grid** — easy, information-rich | **157** ✅ |
| **fusion** | draw **tomorrow's candle** — nearly impossible | **10** ❌ |

**The candle-tomorrow head is too weak an anchor.** Its own numbers say so: `1.141`
against `1.177` for shrugging and drawing the average candle — it beats doing nothing by
**3%**. Tomorrow is genuinely almost unpredictable, so that head yields a feeble
gradient. The model can draw a near-average candle *whatever the token says*, so the
token is never punished for being empty.

TS's codebook stays healthy because its anchor is a task the token **can actually
satisfy**: describe *today*.

Loss re-weighting cannot fix this. Lowering the naming weight (the term that rewards the
cheat) from 1.0 to 0.1 only made accuracy worse — it did not save the dictionary. The
problem is not that the cheat is too rewarded; it is that **being empty is not
punished**.

## The fix to try (after Colab)

Give the token an anchor it can satisfy: **a small head straight off the fused token that
reconstructs TODAY's candle.**

```
                  ┌─ describe TODAY's candle     <- the anchor. Cheap, information-dense,
   token ─────────┤                                 and something the token CAN do.
                  └─ GPT ─┬─ next token
                          └─ TOMORROW's candle   <- the goal
```

Two reasons it should work where the current head does not:

1. **It produces a real gradient.** Describing today is easy, so an empty token is
   immediately and heavily punished — which is exactly what is missing now.
2. **It forces DIRECTION into the token.** The body of a candle is where it closed
   against where it opened. Direction is the one thing we measured *both* halves of the
   tokenizer throwing away (TS keeps `log_return` at 13%; CS scores 8% on direction,
   *below* its luck floor).

This is STORM's reconstruction term — which they weight at **1e-3** against a prediction
loss of 0.1. It was never their objective. It is a **leash**.

It is also NOT the market-rebuilding we already rejected: that asked one token to redraw
thirty companies (~8%, near its information-theoretic ceiling). This asks one token to
describe *the one company's candle today*, which is cheap and which it can do.

## Two other things to remember

**Persistence is a brutal baseline, and it is the honest one.** "Say tomorrow's word is
the same as today's" scores **62-73%**, because market regimes are sticky. Our predictor
has never beaten it. Note that v1's headline result — *"58.6% next-token accuracy"* — was
quoted against a weak *marginal* baseline. It almost certainly never beat persistence
either. **That number must not go in the paper without this floor beside it.**

**What the predictor is actually predicting.** A token is a *regime* label ("a calm
uptrend in a jittery market"), because that is what survived the squeeze. So predicting
the next token is mostly predicting *"the same regime as today"* — which is precisely why
persistence is so hard to beat, and why the candle head matters: it is the only part of
the objective that demands anything about **direction**.
