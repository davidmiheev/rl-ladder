# C10 — A year forward on Bybit, the crash, and data from outside the book

Not a ladder phase: the forward test of the frozen C8 choices and the
lead-lag inputs from the list of next steps after C9.

Writeup: [`docs/writeups/c10-forward.md`](../../docs/writeups/c10-forward.md).

## Files

| file | what |
| --- | --- |
| `bybit.py` | Bybit's free ob200 books and trades → `data/npz/bybit-<contract>_<date>.npz` (RPI prints dropped; book local time = publish + 2 ms) |
| `forward.py` | the frozen configurations on Bybit: the venue-control week and 54 weekly forward sessions; variant sets `stress`, `controls`, `stops`, `stopstress` (`C10_VARSET`) |
| `external.py` | four free outside sources per asset (Binance USDT perp and spot trades, Bybit USDT perp trades and book mid), reduced to `data/c10_ext/` |
| `leadlag.py` | outside features on the 100 ms grid at an arrival delay; `fit` (held-out R², large-move AUC/precision at +0/+100/+250 ms), `models` and `series` (forecasts for the backtest) |
| `llbacktest.py` | the forecasts through the backtest on the C8 OBI configurations of ETHUSDC, XRPUSDC, SUIUSDC |
| `report.py` | every table and figure in the writeup (`docs/assets/c10/`) |

Kernel additions (`experiments/c6_strategy_zoo/zoo.py`, all off by default,
cached results unchanged): `cap_ref` (the position cap in units, valued at the
session's first mid), `cap_resting` (resting orders count against the cap),
`loss_flatten` (the loss limit exits the position at the touch instead of
going reduce-only).

## Commands

    python experiments/c10_forward/forward.py run --shard 0/4         # C10_VARSET=stress|controls|stops|stopstress
    python experiments/c10_forward/forward.py status
    python experiments/c10_forward/leadlag.py prepare --symbol ethusdc --shard 0/4   # needs HF_TOKEN
    python experiments/c10_forward/leadlag.py fit --symbol ethusdc
    python experiments/c10_forward/leadlag.py models --symbol ethusdc
    python experiments/c10_forward/llbacktest.py run --symbol ethusdc --phase main --shard 0/4
    python experiments/c10_forward/llbacktest.py run --symbol ethusdc --phase stress --shard 0/4
    python experiments/c10_forward/llbacktest.py report
    python experiments/c10_forward/report.py

The large USDT contracts (up to 48 million events a day) were converted and
replayed on Kaggle CPU sessions (code embedded in the notebook; results
merged from `c10_days.tgz` into `runs/c8/days/`).

## Results

| run | status | result |
| --- | --- | --- |
| forward, frozen, 6 stresses × 61 sessions × 6 configurations | done | without the 2025-10-10 crash only XRPUSDC → XRPPERP makes money (+3.96%, +0.21% at +2 bp); venue-control week +1.41 to −1.37% against +2.5 to +3.5% on Binance |
| C8's controls on every session | done | none bounds the crash: worst crash-day drawdowns 26–211% |
| stops on every session | done | stop 50 bp: crash drawdowns 0.8–6.4%, forward without the crash −5.59% summed (−8.41% as frozen); picked by the rule fixed in advance |
| stop 50 bp stress tests | done | crash drawdown ≤ 6.4% under every stress; only XRPUSDC → XRPPERP stays positive (+0.32% at +2 bp, +2.09% at latency ×4) |
| lead-lag forecasts, 3 assets × 3 delays | done | 1 s R² roughly doubles with outside data at +0 ms (XRP 6.0 → 8.9%, SUI 2.5 → 6.3%, ETH 3.8 → 8.4%), half gone at +100 ms; large-move precision +2–3 points |
| lead-lag through the backtest | running | |

## Findings

- The C8 edge does not survive the venue: on the same held-out week the six
  configurations make at most +1.41% on Bybit.
- The crash-day losses beyond 100% of capital came from the position cap
  (notional at the current, falling price; resting orders not counted).
- The daily loss limit failed on the crash because reduce-only quoting does
  not fill in a market running away; a stop that takes the touch does.
- Outside markets lead our book by a few hundred milliseconds at most.
