# C7 — Signals and adaptivity

**Goal.** Find out what information and which simple rule C6's strategies were
missing: a better volatility estimate, the book deeper than five ticks,
BTCUSDT's book, and a spread that reacts to the toxicity of our own fills.

Writeup: [`docs/writeups/c7-signals.md`](../../docs/writeups/c7-signals.md).
Not a ladder phase; see [`docs/ladder.md`](../../docs/ladder.md#c7).

## How to run

```bash
# 1. Per-second book features (own assets from data/npz, BTCUSDT streamed)
python experiments/c7_signals/snapshots.py --symbol ethusdc
python experiments/c7_signals/snapshots.py --symbol xrpusdc
HF_TOKEN=... python experiments/c7_signals/snapshots.py --symbol btcusdt --stream

# 2. Volatility: forecast scores, then the strategies with the new sigma
python experiments/c7_signals/vol_bakeoff.py --symbol ethusdc
python experiments/c7_signals/vol_bakeoff.py --symbol xrpusdc
python experiments/c7_signals/run_variants.py --kind vol

# 3. Book information (the LightGBM check needs `uv pip install lightgbm`)
python experiments/c7_signals/book_info.py --symbol ethusdc
python experiments/c7_signals/book_info_gbm.py --symbol ethusdc

# 4. The markout rule: sweep kappa on training days, then held out
python experiments/c7_signals/run_variants.py --kind markout-sweep
python experiments/c7_signals/run_variants.py --kind markout-test
python experiments/c7_signals/run_variants.py --kind markout-breadth

# 5. RL with the extra observations (C6's queue, recipe v2-deep)
python experiments/c6_strategy_zoo/run_queue.py --kind train
python experiments/c6_strategy_zoo/run_queue.py --kind eval

# 6. Figures
python experiments/c7_signals/plot_c7.py
```

The quoter options live in C6's kernel (`zoo.py`) as parameters that default
to C6's behaviour — `vol_scale`, `har_b0..b3`, `markout_kappa`,
`markout_steps`, `markout_halflife` — and `run_baseline.py --set NAME=VALUE`
reaches any of them.

## Results

| run | config | metric | value | run dir |
| --- | --- | --- | --- | --- |
| vol forecast | 11 estimators, 5 min ahead | held-out QLIKE, best vs C6 | ETH 0.320 (HAR) vs 0.367; XRP tie 0.192 | `runs/c7-vol/` |
| vol in strategy | bias-corrected σ | held-out cumulative, better than C6 | 7 / 8 | `runs/c7/*-volscale__*` |
| book information | + deeper levels, OFI, returns, BTC | held-out R² added at 1 s, ETH | +0.2 own book, ≈0 BTC | `runs/c7-book/` |
| markout rule | κ from 4 training days | held-out cumulative, better than C6 | 6 / 6 | `runs/c7/*-mk?__*` |
| markout rule, untuned | κ = 2, seven unseen assets | held-out, better than C6 (excl. AVAX) | 15 / 21 (13 / 18) | `runs/c7/*-mk2__*` |

## Findings

- C6's σ is 1.45× low in variance; the fix is a constant, and it pays.
- The best volatility forecaster is not the best volatility *input*: HAR loses
  to the plain bias correction inside the strategies in 6 of 8 cases.
- Hidden information in the deep book and in BTCUSDT is, at one-second
  resolution and on these assets, too small to change a quote.
- The markout rule is the first change that beats C6's widened baselines on
  held-out data everywhere it was tried.
- Operational: a BTCUSDT session needs 6.3 GB to replay and was killed twice
  alongside training jobs; the raw data is streamed and deleted per day.
