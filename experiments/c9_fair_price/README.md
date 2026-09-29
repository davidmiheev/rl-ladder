# C9 — The fair price, forecasts of the mid, and denoising on every asset

Not a ladder phase. It answers two questions asked after C8 — can the mid (the
price the spread is quoted around) be forecast about a second ahead, and
would quoting around the future mid pay; and does denoising help any
profitable configuration on any asset — and two follow-ups: acting only on
large moves, and DAgger with a teacher that sees the future.

Writeup: [`docs/writeups/c9-fair-price.md`](../../docs/writeups/c9-fair-price.md).

## Setup

C8's month split (24 training sessions, 7 held out) and C8's
(configuration, session) cache, `runs/c8/days/`. Everything is chosen on
training sessions, reported held out and stress-tested with C8's stress set.

The quoting kernel (`experiments/c6_strategy_zoo/zoo.py`) takes an external
series, one row per 100 ms quoting step:

| column / parameter | what it does |
| --- | --- |
| column 0, `fair_mode` 1 | shift the quotes' centre by the value (bp of the mid), on top of the strategy's signal |
| column 0, `fair_mode` 2 | the same, replacing the strategy's signal |
| column 0, `fair_mode` 3 | widen only the side a move of that sign would run over, by the value |
| column 0, `fair_mode` 4 | pull that side |
| `fair_coef` | scale on column 0 |
| columns 1-2, `dag_*` | the DAgger student: a per-side score (market part precomputed, plus the quote's distance, inventory and toxicity) that pulls or widens a side above `dag_thr` |

Off by default; every cached C8 session replays to the identical return and
fill count. The kernel also logs the rule's pre-action bid and ask and the
per-side toxicity.

## How to run

```bash
# 100 ms features on the strategies' own quoting grid (data/c9/<symbol>_<date>.npz)
python experiments/c9_fair_price/features.py --symbol ethusdc --days all

# forecasts of the mid 0.5-3 s ahead, fitted on training sessions, scored held out
python experiments/c9_fair_price/predict.py fit --symbol ethusdc

# the stages (each resumes from the cache; C8_SYMBOLS limits the assets)
python experiments/c9_fair_price/c9.py run --stage denoise --shard 0/2
python experiments/c9_fair_price/c9.py status
#   denoise, denoise-E       five denoisers on OBI x 9 assets and BNB GLFT, and their stress tests
#   oracle                   quotes centred on the future mid (0.5-3 s), and noisy oracles of set R^2
#   predict, predict-E       micro-price, fitted micro-price, linear and boosted-tree fair prices
#   largemove, largemove-E   widen or pull before large moves: oracle, and gated forecasts
#   dagger, dagger-E         the DAgger students' threshold and action, and their stress tests

# DAgger rounds (rollouts on training sessions, teacher labels, refit)
C8_SYMBOLS=ethusdc python experiments/c9_fair_price/dagger.py rollout --round 0 --shard 0/2
C8_SYMBOLS=ethusdc python experiments/c9_fair_price/dagger.py fit --round 0
C9_TEACHER=pull:30:0.99 ...                  # the second teacher (see below)

# tables and figures (docs/assets/c9/)
python experiments/c9_fair_price/report.py
```

## Results

Held-out cumulative return, % of book, over the seven held-out sessions
unless marked; full tables in the writeup (`report.py` prints them).

**Forecasting the mid** (best held-out R², boosted trees on 17 causal
features): 1 s ahead, XRPUSDT 10.2%, XRPUSDC 6.0%, ETHUSDC 3.9%, BNBUSDC 3.2%,
SUIUSDC 2.5%, AVAXUSDC 1.8%; lower at 1.5 and 3 s. OBI's own signal: ≈ 0.

**Quoting around the future mid** (C8 choice → mid 1 s ahead → 3 s ahead):

| configuration | C8 choice | +1 s | +3 s |
| --- | --- | --- | --- |
| ETHUSDC GLFT | −3.25 | +4.30 | +19.67 |
| ETHUSDC OBI | +0.85 | +2.41 | +16.65 |
| BNBUSDC GLFT | +1.72 | +14.72 | +41.33 |
| AVAXUSDC OBI | +2.68 | +11.95 | +15.98 |
| XRPUSDC OBI | +2.84 | +3.16 | +3.82 |

A noisy oracle needs R² ≈ 0.4 to beat the C8 choice on both halves of ETH
OBI; below 0.2 it mostly loses.

**Causal fair prices**: only AVAXUSDC OBI improves (boosted trees, +3.33
against +2.68, survives the stress set); XRPUSDC gains 0.4 points from the
micro-price, all of it the crash; everything else adopted on training is
worse held out.

**Large moves**: an oracle that pulls a quote before the top 1% of 3 s moves
gives ETH OBI +6.53 and ETH GLFT +2.05; the forecast rules chosen on training
give −0.10 (ETH OBI), +2.38 (XRP OBI, C8 +2.84) and −2.20 (ETH GLFT).

**DAgger** (ETH, chosen on training sessions): first teacher (5 s markout)
ETH GLFT −1.05, ETH OBI +0.91; second teacher (the large-move oracle, AUC
0.71) ETH GLFT −1.12, ETH OBI +0.52 (below the rule on training, so not
adopted). C8 choices: −3.25 and +0.85.

**Denoising, 9 assets**: no denoiser both improves a profitable
configuration held out and survives the stress tests; SUIUSDC OBI loses 1–3
points held out with every denoiser that helps it on training.

## Findings

- The fair price matters enormously if it is right, and cannot be forecast
  well enough to be: the real forecasts reach R² 0.02–0.10 against the ~0.4
  that pays.
- The value is in rare large moves the book does not anticipate at 1–3 s:
  the forecasts' largest calls are ~0.6 bp against 3–5.5 bp real large
  moves.
- The first DAgger teacher was misaligned (a quote the opposite touch
  reaches is mostly one the price runs through: 58–69% "losing" fills); the
  second teacher is the large-move oracle policy. Both students halve ETH
  GLFT's loss and neither beats the rule on ETH OBI: the large-move student
  ranks the dangerous moments (AUC 0.71) but is right on 3–11% of its pulls.
- Denoising is done: the smoothing that helps is inside the markout rule and
  the circuit breaker already.
