# Bubble Bi

### Reading the stock market like a language

**Open [`Bubble_Bi.ipynb`](Bubble_Bi.ipynb). That is the project.**

Run it top to bottom. Every section explains what is about to happen in plain language,
does it, then **proves** it worked and tells you what you now have. You do not need to
read any code to follow it.

---

## The idea

Every day a stock leaves a trace: open, high, low, close, volume. Markets repeat certain
moods — calm drift, panic, sharp reversal — but nobody labels them, and there is no
dictionary of them.

So we let the machine build the dictionary itself. Each stock-day is squeezed into a
single **token** — one word out of 512 that the machine invents on its own. A company's
history then becomes a sentence:

```
AAPL:  … #147  #147  #391  #208  #208  #63  →  what comes next?
```

From there it is the same trick as ChatGPT: read the sentence, predict the next word.

```
TS encoder (one company, over time)  ──┐
                                       ├─► cross-attention ─► ONE codebook ─► GPT
CS encoder (the whole market, on a day)┘
```

Inspired by **STORM** (Zhao et al., WSDM '26 — the PDF is in this repo), with one
deliberate deviation: STORM's tokens feed a linear factor model that predicts returns.
Ours feed a **Llama-3-style Transformer** that predicts the next *token* and the next
*candle* — a world model, not a factor model.

## The one rule

**Nothing may ever look into the future.** Break it and the model looks brilliant while
losing real money. It is the most common way a financial model fools the person who built
it.

So the notebook does not *assert* that its features are backward-looking — it **proves**
it, on the real data, every run: take the data, delete the future, recompute everything,
and check that not a single past value changed. The leak detector is itself tested by
planting a leak and confirming it gets caught.

## Where things are

| | |
|---|---|
| `Bubble_Bi.ipynb` | **the project** — the guided tour you actually run |
| `bubble_bi/` | the code the notebook calls |
| `tests/` | 178 tests |
| `docs/` | open questions, written down honestly |
| `STORM.pdf` | the paper |

## Running it

Locally, a short run finishes on a laptop CPU. For anything you intend to believe, use a
GPU:

**On Colab:** *Runtime → GPU*, mount Drive, point `data_dir` at it, and raise `steps`.
Models are saved as they finish and reloaded on re-run, so a dropped session costs
nothing.

## Honest status

| | |
|---|---|
| data pipeline, no-lookahead proof | ✅ works |
| **TS** tokenizer (one company) | ✅ works — perplexity 157, explains 43% of a held-out day |
| **CS** tokenizer (whole market) | ✅ works — its words explain **88%** of how violent the market was, and **nothing** about which way it went |
| fusion + Llama-3 predictor | ⚠️ **built, and it fails** — see below |
| the trading agent | ⬜ not built. No backtest, no costs, no evidence of profit. |

The predictor **loses to persistence** ("tomorrow's word is the same as today's") and its
codebook **collapses** to ~12 words of 512. This is diagnosed, not fixed:
[`docs/OPEN-QUESTION-codebook-collapse.md`](docs/OPEN-QUESTION-codebook-collapse.md).
The notebook shows the ❌ and says *"do not pretend this passed"* rather than hiding it.

## The finding the project turns on

**Given the candle — explicitly, as four features — the tokenizer threw it away.**

```
volatility  50%      macd 85%   rsi 75%   sma_ratio_20 73%
memory      49%
price       45%      ...against...
candle       3%      gap 4%   body 5%   log_return 9%   lower_wick -1%
```

And CS said the same about the market: its words explain **88%** of how *violent* a day
was, and **8%** of which *way* it went — *below* its own shuffled floor.

It is right to. One token is **9 bits**. Fifteen days of MACD is a smooth curve; fifteen
candle bodies are fifteen independent random numbers. A 9-bit code spends itself on what
compresses. **This is not the model failing — it is the model reporting that daily
direction is close to noise and volatility regime is not.**

So we do **not** force it to keep the candle. If direction turns out not to be there, the
*downstream task* changes to fit what the model can actually see — trading the **regime**
rather than the direction. See
[`docs/DECISION-let-the-model-choose.md`](docs/DECISION-let-the-model-choose.md).

**A number that must not be quoted alone:** the previous version of this project reported
*"58.6% next-token accuracy"* against a weak baseline. Against **persistence** — the
honest floor — it would not have cleared the bar either.

## The previous version

v1 (a complete, tested D=22 pipeline, M0–M3) is preserved at the tag
**`v1-d22-frozen`**. This project is a fresh rewrite, not a refactor.

```bash
git checkout v1-d22-frozen
```
