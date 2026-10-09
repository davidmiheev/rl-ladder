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

### Stage 1 for the six added contracts, and how much of each leader is the rebate

Every backtest assumes Binance's best market-maker rebate (maker fee −0.5 bp,
`config.MAKER_FEE`). Bybit's standard maker fee is +2 bp. A session's rebate
income is its turnover x 0.5 bp of capital, so high-turnover settings can rank
first on the rebate alone. Screen leaders (24 sessions), with the return
recomputed without the rebate and at Bybit's standard fee:

| contract | leader | screen | rebate part | no rebate | +2 bp (Bybit VIP 0) | turnover/session |
| --- | --- | --- | --- | --- | --- | --- |
| XRPPERP | OBI 8 bp, κ 2 | +7.65% | +1.65 | +5.99% | −0.39% | 7.4x |
| SUIPERP | hfgrid 8 bp | +2.57% | +0.16 | +2.41% | +1.77% | 1.1x |
| XRPUSDT | OBI 8 bp, κ 4 | +6.65% | +1.32 | +5.32% | +0.18% | 7.3x |
| SUIUSDT | hfgrid 4 bp | +9.01% | +0.74 | +8.27% | +5.34% | 5.8x |
| AVAXUSDT | OBI 6 bp, κ 4 | +5.05% | +2.38 | +2.67% | −6.35% | 11.8x |
| ENAUSDT | GLFT 8 bp, κ 4 | +61.58% | +111.41 | **−49.84%** | −100% | 266.7x |
| BNBUSDT | GLFT 4 bp | +5.86% | +1.14 | +4.72% | +0.26% | 6.4x |
| DOGEUSDT | OBI 16 bp, κ 4 | +2.72% | +0.23 | +2.49% | +1.57% | 1.5x |
| ADAUSDT | hfgrid 4 bp | +6.64% | +0.48 | +6.17% | +4.28% | 3.7x |
| LINKUSDT | hfgrid 4 bp, κ 2 | +6.68% | +0.19 | +6.49% | +5.75% | 1.1x |
| LTCUSDT | hfgrid 4 bp | +8.51% | +0.72 | +7.79% | +4.97% | 5.6x |

(No rebate = return − turnover x 0.5 bp per session, chained; the fee stress
runs in stage 5 recompute it in the backtest.) ENAUSDT's top three all turn
over 270-310x capital a session and lose 47-54% without the rebate: the
rebate is the whole result. Its families stage runs last. The slow grid
quoters (hfgrid) keep most of their return without the rebate.

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

### AVAXUSDT: validation and test

| entry | training (48) | validation (27) | test (27) | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, all sessions: OBI 6 bp, κ 4 + momentum (50 s) | +249.50% | +1.37% | **−1.26%** | −2.50% | −2.47% | 12/27 | 132 |
| pick, crash-guarded: OBI 6 bp, κ 4 + markout half-life 5 s | +17.84% | +1.25% | **−1.69%** | −2.62% | −2.39% | 15/27 | 52 |
| C8's choice, unchanged (AVAXUSDC OBI) | −79.64% | −1.61% | **+2.64%** | +1.02% | +2.19% | 16/27 | 49 |

The crash-guarded pick is built on the screen leader, whose crash-session
drawdown was 31% (see the crash check above); the guard only applied to the
added markout half-life.

Stress set on the test block (chained return, 27 sessions):

| stress | pick, all sessions | pick, crash-guarded |
| --- | --- | --- |
| as chosen | −1.26% | −1.69% |
| maker fee 0 (no rebate) | −2.47% | −2.39% |
| maker fee +1 bp | −4.85% | −3.76% |
| maker fee +2 bp (Bybit VIP 0) | −7.17% | −5.12% |
| latency x2 | +2.88% | −0.86% |
| latency x4 | −2.82% | +2.57% |
| queue risk-averse | −1.80% | −1.82% |
| queue log | −0.65% | −2.71% |
| size x2 | −1.77% | −0.97% |
| floor x0.8 | −0.07% | +0.83% |
| floor x1.25 | −1.61% | +0.99% |
| κ x0.5 | −5.04% | −5.85% |
| κ x2 | +2.16% | +1.20% |

**AVAXUSDT verdict.** Both picks lost on the test (−1.26%, −1.69%) and lose
3.8-7.2% at Bybit's real fee levels (+1 to +2 bp). The stresses scatter on
both sides of zero (latency x2 +2.88% vs latency x4 −2.82% for the same
configuration), which is what no edge looks like. C8's Binance choice, run
unchanged, made +2.64% on the same test sessions (+2.19% without the rebate),
after −79.6% on training (the crash) and −1.6% on validation: it was not
selectable on anything before the test.

So far (SUIPERP, AVAXUSDT) the full procedure on a year of Bybit has not
produced a configuration that beats C8's untouched Binance choice on the
locked test.

## Status

See `python experiments/c13_bybit_contest/contest.py status`. Kaggle runs use
`units` dumps and 3 workers per notebook (4 workers ran out of memory on
XRPUSDT sessions).
