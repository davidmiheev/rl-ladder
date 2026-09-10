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

The data Part C actually runs on is the private Hugging Face dataset
[`cardinalai/backtest_data`](https://huggingface.co/datasets/cardinalai/backtest_data):
Binance USDⓈ-M perpetual feeds already normalised into the format above, one
`.npz` per symbol per day, 2025-08-26 to 2025-09-25, for ETH/BTC/XRP/SUI/BNB/
AVAX/ENA against both USDT and USDC. Access needs a token with read rights on
that dataset:

```bash
export HF_TOKEN=...            # never written into the repo; data/ is gitignored
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("cardinalai/backtest_data", "ethusdc_2025-08-26.npz",
                repo_type="dataset", local_dir="data/npz")
EOF
```

Each day's file opens with a depth-clear plus a full book snapshot inside its
first second, so days are self-contained: no external end-of-previous-day
snapshot is needed and any single day can be replayed on its own. (Verify this
before assuming it of a new symbol — it is a property of how this feed was
recorded, not of the format.)

Two fallbacks, both documented in hftbacktest's `examples/Data Preparation.ipynb`,
if that dataset is unavailable:

1. **Tardis.dev sample data** — free single-day samples of Binance futures
   incremental L2 book and trades (`datasets.tardis.dev/v1/binance-futures/...`).
2. **hftbacktest's own collector** (`companions/hftbacktest/collector/`) — a Rust
   feed recorder you run yourself to accumulate days.

Either has to go through the Data Preparation notebook to reach the normalized
format.

## Order latency

The dataset carries no recorded order latency, so C1 synthesises it from the
feed latency the data already contains (`local_ts - exch_ts`), following
hftbacktest's [Order Latency Data](https://hftbacktest.readthedocs.io/en/latest/tutorials/Order%20Latency%20Data.html)
tutorial: entry latency is 4x the feed latency and response latency 3x, sampled
once a second and interpolated between. On these days that is a round trip of
roughly 20 ms, which says the feed was recorded from outside the exchange's
colocation. Every Part C result is conditional on that assumption; it is the
single most load-bearing modelling choice in the whole of Part C, because a
market maker's fills live and die by whether its cancel arrives before the
trade does.

`experiments/c1_glft_baseline/config.py` writes the generated files to
`data/latency/`, and `run_calibrate.py --verify-latency DATE` checks the
generator against hftbacktest's own helper (they agree bit for bit; ours exists
only because the packaged one copies a whole day through polars and does not
fit in memory).

## The split rule

Decided **once**, in C1, and recorded in
[`experiments/c1_glft_baseline/config.py`](../experiments/c1_glft_baseline/config.py),
which is where every Part C script reads it from so it cannot drift:

- **train** — 2025-08-26 to 2025-09-14. Calibration, sweeps, and (later) RL
  training all live here.
- **held out** — 2025-09-15 to 2025-09-25. Read once, at the end of a phase.

The split is chronological because the data is a time series: a random split of
days would leak the regime a policy is about to be tested in. C3, C4 and C5 all
reuse it unchanged. Every Part C claim ("RL beats the calibrated formula") is a claim
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
