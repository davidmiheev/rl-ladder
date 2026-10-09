# C12 — OBI re-chosen on Bybit's own study month

Not a ladder phase. C10 ran C8's Binance-chosen configurations on Bybit; C12
asks whether choosing again on Bybit helps. OBI floor (4-24 bp) x κ (C8's
grid), with the 50 bp stop, chosen on Bybit's 24 study-month sessions
(c8.TRAIN dates) and checked on the same held-out week C10 used
(2025-09-19 .. 09-25, 7 sessions).

| file | what |
| --- | --- |
| `reselect.py` | grid, pick, phases `train`, `test`, `train-long`, `test-long` |

## Result

| contract | re-chosen (floor, κ) | training | held-out week | C8's choice + 50 bp stop, held-out week |
| --- | --- | --- | --- | --- |
| XRPPERP | 8 bp, κ 4 | +1.76% | −0.28% (145 fills) | +0.01% (67 fills) |
| XRPUSDT | 12 bp, κ 2 | +0.69% | −1.70% (103) | −0.02% (276) |
| SUIPERP | 6 bp, κ 4 | +3.20% | −1.88% (147) | −0.72% (220) |
| SUIUSDT | 8 bp, κ 2 | +7.45% | **+2.29%** (534) | −1.28% (24) |
| AVAXUSDT | 16 bp, κ 2 | +4.10% | −2.52% (36) | −1.01% (84) |

Re-choosing beat C8's choice on 1 of 5 contracts. One month of training is
too short on Bybit too; C13 (`experiments/c13_bybit_contest`) replaces this
with six months of training, a validation block and a locked test.
