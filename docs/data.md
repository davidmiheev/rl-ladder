# Data (Part C)

Part C replays real order-book data through hftbacktest. None of it lives in this
repo — `data/` is gitignored. This page records where it comes from and what
shape it has to be in.

## Format

hftbacktest digests a numpy structured array with 8 fields, one row per market
event (see `companions/hftbacktest/docs/data.rst` for the authoritative table):

| field | type | meaning |
| --- | --- | --- |
| `ev` | u64 | event flags (BUY/SELL, DEPTH/TRADE, EXCH/LOCAL) |
| `exch_ts` | i64 | exchange timestamp, ns |
| `local_ts` | i64 | local receive timestamp, ns — the latency the backtest models |
| `px` | f64 | price |
| `qty` | f64 | quantity |
| `order_id` | u64 | L3 market-by-order feeds only |
| `ival` / `fval` | i64 / f64 | reserved extras |

The two timestamps are the point of the format: the gap between them is the feed
latency your strategy actually experiences, and it is what makes a backtest
result about market making rather than about clairvoyance.

## Getting data

Two supported paths, both documented in hftbacktest's
`examples/Data Preparation.ipynb`:

1. **Tardis.dev sample data** — free single-day samples of Binance futures
   incremental L2 book and trades (`datasets.tardis.dev/v1/binance-futures/...`).
   Enough to reproduce C1 and to develop C2's environment wrapper against.
2. **hftbacktest's own collector** (`companions/hftbacktest/collector/`) — a Rust
   feed recorder you run yourself to accumulate days. This is the path to a real
   train/held-out split, since one sample day cannot support one.

Convert either into the normalized `.npz` format with the Data Preparation
notebook before use.

## Layout

```
data/
├── raw/<exchange>/<symbol>/<YYYYMMDD>.csv.gz     as downloaded/collected
├── npz/<exchange>/<symbol>/<YYYYMMDD>.npz        normalized, what backtests read
└── latency/<exchange>/<YYYYMMDD>.npz             order-latency data, if collected
```

## The split rule

Decide the train/held-out day split **once**, in C1, and record the exact dates
in `experiments/c1_glft_baseline/README.md`. C3, C4 and C5 all reuse it
unchanged. Every Part C claim ("RL beats the calibrated formula") is a claim
about held-out days; re-picking the split after seeing results makes the number
meaningless, and C4's third item exists specifically to check for that failure.

Aim for enough held-out days to span more than one volatility regime. A policy
that beats GLFT on three quiet days has demonstrated nothing about the days
market making actually gets tested.

## Latency and fill realism

Before trusting any C3–C5 result, read hftbacktest's
`Impact of Order Latency.ipynb` and `Probability Queue Models.ipynb`. An RL
policy will happily exploit an optimistic queue model — a "learned improvement"
that is really the fill simulator being too generous is the most likely wrong
result in Part C, and it looks exactly like success.
