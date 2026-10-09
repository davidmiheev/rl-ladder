# C13 — C8's selection procedure on a year of Bybit, as a heuristic contest

Not a ladder phase. C10 found that C8's choices, made on 24 Binance sessions,
did not travel to a year of Bybit; C12 found that re-choosing on Bybit's own
24-session study month did not help either. C13 reruns C8's whole procedure
on Bybit with far more sessions and every heuristic the project has as a
contestant, and scores it on blocks it never chose on.

## Files

| file | what |
| --- | --- |
| `contest.py` | the split, the five stages, the two selection rules; `run`, `units` (pending work for a fresh Kaggle session), `status`, `report` |

## Split (sessions every 4th day, blocks never mix)

| block | dates | sessions | used for |
| --- | --- | --- | --- |
| screen | 2025-08-26 .. 2026-02-26, every 8th day | 24 | stage 1 |
| training | 2025-08-26 .. 2026-02-28, plus the 2025-10-10 crash | 48 | stages 2-3, the final pick |
| validation | 2026-03-02 .. 2026-06-14 | 27 | the final pick |
| test | 2026-06-18 .. 2026-10-02 | 27 | reported once, never chosen on |

## Stages

1. **screen**: OBI (floor 4-24 bp x κ 0/2/4 x skew 14.8/29.6 bp), GLFT and
   hfgrid (floor 4/8/16 bp x κ) — 54 settings per contract.
2. **families**: on top of the screen leader, every C8 family (breaker, loss
   limit, smoothing, markout half-life, momentum, per-side markout, requote
   hysteresis, toxicity sizing, inventory flatten, time of day, alpha
   clipping) and C10's (stop, units cap), on all 48 training sessions.
3. **greedy**: C8's forward selection over the families' best variants.
4. **validation**: the screen's top 3 and each rule's family bests and
   greedy result; each rule picks the best on training + validation.
5. **test**: both picks and C8's stress set on the test block.

**Two selection rules.** C8's rule ranks by the chained return over all
training sessions. The 2025-10-10 crash is one of them, and on XRPPERP it
alone decided the ranking (the best breaker: +46.6% on the crash, −8.9% over
the other 47). So stages 2-4 also run under C10's crash-guarded rule
(crash-session drawdown at most 10%, then the best chained return without
the crash), and both picks go to the test.

Contracts: XRPPERP, SUIPERP, XRPUSDT, SUIUSDT, AVAXUSDT (C10's five), then
ENAUSDT, BNBUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, LTCUSDT.

## Results so far

### Stage 1 (screen, 24 sessions; in-sample: the best of 54 on the sessions it was chosen on)

| contract | leader | return | fills/session | C8's choice on the same sessions |
| --- | --- | --- | --- | --- |
| XRPPERP | OBI 8 bp, κ 2 | +7.65% | 148 | +2.49% (56 fills) |
| XRPUSDT | OBI 8 bp, κ 4 | +6.65% | 146 | – |
| SUIUSDT | hfgrid 4 bp | +9.01% | 117 | – |
| AVAXUSDT | OBI 6 bp, κ 4 | +5.05% | 236 | – |
| SUIPERP | hfgrid 8 bp | +2.57% | 22 | – |

### Stages 2-3, training (48 sessions)

| contract, rule | added | all 48 | crash alone | other 47 | crash-day drawdown | fills |
| --- | --- | --- | --- | --- | --- | --- |
| XRPPERP, leader alone | – | −7.92% | −9.79% | +2.07% | – | 164 |
| XRPPERP, all sessions (round 3) | breaker + momentum + per-side markout | +109.54% | +128.77% | −8.41% | 33.9% | 135 |
| XRPPERP, crash-guarded (round 3) | 100 bp stop + signal smoothing | +6.63% | −1.06% | +7.77% | 3.6% | 66 |
| SUIPERP, all sessions | momentum + breaker | +18.85% | +10.27% | +7.78% | 24.9% | 30 |
| SUIPERP, crash-guarded | nothing beats the leader | +6.70% | −1.30% | +8.10% | 7.2% | 18 |

### Crash-session check (2025-10-10)

Under C8's rule every greedy result so far wins the crash session by a wide
margin. The per-session records show how: during the crash the position runs
far past its cap (the cap applies when quotes are placed; a sweep that fills
several resting orders at once overshoots it).

| contract | entry | crash session | drawdown | peak position / cap | peak position value | fills |
| --- | --- | --- | --- | --- | --- | --- |
| XRPPERP | stage 1 leader | −9.79% | 56.7% | 1.53 | $141,663 | 1177 |
| XRPPERP | greedy, all sessions | +126.96% | 34.7% | 2.19 | $216,589 | 1535 |
| XRPPERP | greedy, crash-guarded | −1.15% | 2.0% | 0.59 | $58,995 | 65 |
| XRPUSDT | greedy, all sessions | +107.72% | 28.8% | 2.19 | $218,868 | 2402 |
| SUIUSDT | greedy, all sessions | +102.98% | 14.6% | 1.03 | $102,476 | 840 |
| AVAXUSDT | stage 1 leader | +25.09% | 31.1% | 1.52 | $93,046 | 962 |
| AVAXUSDT | greedy, all sessions (momentum) | +290.11% | 41.2% | 2.63 | $260,516 | 2789 |

So C8's rule rewards configurations that take on up to 2.6x the capital in
inventory during the crash and happen to be on the right side of the
rebound, while losing over the other 47 sessions (AVAXUSDT: −10.4%,
XRPPERP greedy: +3.2%). These crash-day numbers depend on fills at wick
prices in a book that was thin and fast. Under the crash-guarded rule
(drawdown at most 10%) none of these configurations is eligible.

One gap in the crash-guarded rule: stage 3 starts from the screen leader,
and the guard applies only to what is added. On AVAXUSDT no family variant
passes the guard, so the crash-guarded "pick" is the screen leader itself,
whose crash drawdown is 31.1%. Its test result is reported with that
caveat, and it gets the "without the crash" stress like every pick.

### SUIPERP: validation and test (first contract through all five stages)

| entry | training (48) | validation (27) | test (27) | test without its best session | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- |
| pick, all sessions: hfgrid 8 bp + momentum + breaker | +18.85% | +1.21% | **−1.01%** | −1.64% | 10/27 | 11 |
| pick, crash-guarded: hfgrid 8 bp + momentum (10 s) | +5.79% | +2.20% | **+0.22%** | −1.60% | 11/27 | 11 |
| C8's choice, unchanged (SUIUSDC OBI) | +3.10% | +2.16% | **+1.68%** | −1.45% | 14/27 | 15 |

Stress set on the test block (chained return, 27 sessions):

| stress | pick, all sessions | pick, crash-guarded |
| --- | --- | --- |
| as chosen | −1.01% | +0.22% |
| maker fee 0 (no rebate) | −1.18% | +0.05% |
| maker fee +1 bp | −1.52% | −0.29% |
| maker fee +2 bp | −1.86% | −0.63% |
| latency x2 | −0.20% | +0.12% |
| latency x4 | −0.51% | −1.57% |
| queue risk-averse | −0.80% | +0.26% |
| queue log | −0.82% | +0.23% |
| size x2 | −1.39% | −0.03% |
| floor x0.8 | −0.77% | −0.14% |
| floor x1.25 | −0.65% | +0.14% |

**SUIPERP verdict.** The procedure did not find an edge on SUIPERP. C8's
rule picked the training winner (+18.85%, most of it momentum on the crash
session), which lost on the test (−1.01%) and under every stress. The
crash-guarded pick held at about zero (+0.22%) and turned negative with no
rebate plus 1 bp, or at 4x latency. Neither beat C8's Binance choice run
unchanged (+1.68%). Without each entry's best test session all three are
negative (−1.45% to −1.64%). The hfgrid quoters trade about 11 times a session,
so these are small samples.

## Status

See `python experiments/c13_bybit_contest/contest.py status`. Kaggle runs use
`units` dumps and 3 workers per notebook (4 workers ran out of memory on
XRPUSDT sessions).
