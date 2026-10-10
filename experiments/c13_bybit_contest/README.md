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

### Stages 2-3, training (48 sessions) — interim snapshot after greedy round 3; final results in `report.py` and the writeup

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

### Stage 2 for BNB, DOGE, ADA, LINK and LTC (training, 48 sessions)

| contract | entry | all 48 | crash alone | other 47 | without the rebate | fills/session |
| --- | --- | --- | --- | --- | --- | --- |
| BNBUSDT | leader: GLFT 4 bp | +33.02% | +22.43% | +8.65% | +29.01% | 120 |
| | best, all sessions: + momentum | +46.15% | +42.19% | +2.79% | +40.09% | 142 |
| | best, crash-guarded: + breaker | +3.70% | −5.16% | +9.35% | +1.66% | 84 |
| DOGEUSDT | leader: OBI 16 bp, κ 4 | +13.28% | +11.11% | +1.96% | +12.75% | 32 |
| | best, all sessions: + momentum | +88.28% | +85.14% | +1.70% | +86.52% | 39 |
| | best, crash-guarded: + loss limit | −1.69% | −5.05% | +3.54% | −2.15% | 30 |
| ADAUSDT | leader: hfgrid 4 bp | +8.29% | −0.77% | +9.13% | +7.32% | 72 |
| | best, all sessions: + momentum | +46.51% | +70.67% | −14.16% | +37.84% | 482 |
| | best, crash-guarded: + loss limit | +5.13% | −3.60% | +9.05% | +4.20% | 70 |
| LINKUSDT | leader: hfgrid 4 bp, κ 2 | +8.60% | +3.42% | +5.01% | +8.22% | 25 |
| | best, all sessions: + momentum | +20.01% | +27.47% | −5.85% | +17.14% | 152 |
| | best, crash-guarded: + signal smoothing | +11.70% | +3.42% | +8.01% | +11.32% | 26 |
| LTCUSDT | leader: hfgrid 4 bp | +5.67% | −5.72% | +12.09% | +4.15% | 117 |
| | best, all sessions: + momentum | +34.71% | +58.69% | −15.11% | +25.95% | 538 |
| | best, crash-guarded: + loss limit | +8.09% | −3.45% | +11.95% | +6.54% | 117 |

Under C8's rule momentum wins on all five, each time on the crash session
(+27% to +85% that day) while making −15% to +3% over the other 47 — the
same pattern as the original five. Under the crash-guarded rule the bests
are the protective families (breaker, loss limit, smoothing), at +3.5% to
+12% over the 47 non-crash sessions.

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
and the guard applies only to what is added. On AVAXUSDT the crash-guarded
greedy result added nothing, so it is the screen leader itself, whose crash
drawdown is 31.1%. (Correction, added later: the final crash-guarded pick on
AVAXUSDT is a markout half-life variant whose crash-session drawdown is
9.3%, so it does pass the guard; see the scoreboard notes.)

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

The crash-guarded pick's crash-session drawdown is 9.3%, inside the guard
(an earlier note here said the guard did not cover it; it does).

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

### XRPPERP: validation and test

| entry | training (48) | validation (27) | test (27) | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, all sessions: OBI 8 bp, κ 2 + breaker, momentum, per-side markout, stop, requote hysteresis, markout half-life | +134.30% | +0.03% | **−4.54%** | −4.90% | −4.94% | 10/27 | 54 |
| pick, crash-guarded: OBI 8 bp, κ 2 + 100 bp stop, smoothing, alpha clip, units cap, time of day, inventory flatten | +12.11% | +0.63% | **+1.22%** | −0.05% | +0.86% | 11/27 | 51 |
| C8's choice, unchanged (XRPUSDC OBI) | +4.15% | −0.97% | **+1.37%** | +0.78% | +1.09% | 16/27 | 38 |

| stress (test block) | pick, all sessions | pick, crash-guarded |
| --- | --- | --- |
| as chosen | −4.54% | +1.22% |
| maker fee 0 (no rebate) | −4.88% | +0.86% |
| maker fee +1 bp | −5.43% | +0.14% |
| maker fee +2 bp (Bybit VIP 0) | −6.01% | −0.57% |
| latency x2 | −5.57% | +1.51% |
| latency x4 | −5.85% | +2.89% |
| queue risk-averse | −3.99% | +1.86% |
| queue log | −4.16% | +0.79% |
| size x2 | −8.49% | +0.98% |
| floor x0.8 | −4.56% | +1.73% |
| floor x1.25 | −0.36% | +1.25% |
| κ x0.5 | −5.08% | +2.59% |
| κ x2 | −4.47% | −0.05% |

**XRPPERP verdict.** C8's rule picked the crash-day winner (+134% on
training, of which +127% the crash) and it lost under every stress on the
test (−0.4% to −8.5%). The crash-guarded pick is the first contest entry that
holds up: +1.22% on the test, positive in 10 of 12 stresses, worst session
drawdown 1.3%. It is not better than C8's untouched choice on the same
sessions (+1.37%), and without its best test session it is flat (−0.05%).
It loses at Bybit's standard +2 bp maker fee (−0.57%).

### XRPUSDT: validation and test

| entry | training (48) | validation (27) | test (27) | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, all sessions: OBI 8 bp, κ 4 + momentum, time of day | +135.05% | −0.65% | **−0.18%** | −2.38% | −1.41% | 12/27 | 100 |
| pick, crash-guarded: OBI 8 bp, κ 4 + inventory flatten, units cap | +7.82% | +0.09% | **−1.01%** | −1.78% | −2.24% | 11/27 | 133 |
| C8's choice, unchanged (XRPUSDT OBI) | +36.61% | +0.33% | **−0.40%** | −1.30% | −2.75% | 12/27 | 216 |

| stress (test block) | pick, all sessions | pick, crash-guarded |
| --- | --- | --- |
| as chosen | −0.18% | −1.01% |
| maker fee 0 (no rebate) | −1.41% | −2.24% |
| maker fee +1 bp | −3.81% | −4.66% |
| maker fee +2 bp (Bybit VIP 0) | −6.17% | −7.01% |
| latency x2 | +1.35% | +3.38% |
| latency x4 | +0.55% | +2.02% |
| queue risk-averse | −0.45% | +1.43% |
| queue log | −0.34% | +1.42% |
| size x2 | +2.99% | −2.11% |
| floor x0.8 | +1.08% | −2.85% |
| floor x1.25 | +2.43% | +2.02% |
| κ x0.5 | −1.05% | −1.56% |
| κ x2 | +0.86% | +2.30% |

**XRPUSDT verdict.** All three entries are within a point of zero on the test
and all lose at Bybit's real fees (−3.8% to −7.0% at +1 to +2 bp). The
stresses scatter on both sides of zero. No edge.

### SUIUSDT: validation and test

| entry | training (48) | validation (27) | test (27) | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, all sessions: hfgrid 4 bp + momentum, breaker, toxicity sizing, markout half-life | +107.68% | −7.50% | **−4.81%** | −5.40% | −6.56% | 7/27 | 613 |
| pick, crash-guarded: hfgrid 4 bp + breaker (5-min cooldown, vol 2.5) | +23.07% | +0.96% | **+4.60%** | +2.76% | +3.33% | 14/27 | 122 |
| C8's choice, unchanged (SUIUSDT OBI) | −145.45% | −0.63% | **−4.64%** | −5.31% | −4.83% | 13/27 | 25 |
| C8's choice, unchanged (SUIUSDC OBI) | +36.15% | −2.74% | **−4.99%** | −6.27% | −5.52% | 12/27 | 77 |

| stress (test block) | pick, all sessions | pick, crash-guarded |
| --- | --- | --- |
| as chosen | −4.81% | +4.60% |
| maker fee 0 (no rebate) | −6.56% | +3.33% |
| maker fee +1 bp | −9.95% | +0.84% |
| maker fee +2 bp (Bybit VIP 0) | −13.23% | −1.60% |
| latency x2 | −4.64% | +4.06% |
| latency x4 | −4.52% | +3.82% |
| queue risk-averse | −5.96% | +4.47% |
| queue log | −5.71% | +3.86% |
| size x2 | −8.34% | +5.15% |
| floor x0.8 | −4.59% | +6.14% |
| floor x1.25 | −5.14% | +5.04% |

**SUIUSDT verdict.** The first contest pick that beats C8's untouched
choices on the locked test, and by a margin: the crash-guarded pick (the
grid quoter with a volatility breaker) makes +4.60% against −4.64% and
−4.99%, stays positive without its best session (+2.76%) and without the
rebate (+3.33%), and is positive in 9 of 10 stresses. It fails only at
Bybit's standard +2 bp maker fee (−1.60%); at +1 bp it keeps +0.84%. Worst
session drawdown 2.3%, about 120 fills a session. C8's rule picked the
high-turnover crash winner (+107.7% on training, 856 fills a session), which
lost under every stress.

### ENAUSDT: validation and test (a backtest artefact)

Both rules picked the same configuration (GLFT 8 bp + stop, momentum,
per-side markout, 25 bp loss limit with flatten).

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (stop, momentum, per-side markout, loss limit) | +369.67% | +11024.06% | **-6.96%** | -6.73% | -10.02% | 0/27 | 415 |
| pick, crash-guarded: greedy (stop, momentum, per-side markout, loss limit) | +369.67% | +11024.06% | **-6.96%** | -6.73% | -10.02% | 0/27 | 415 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -6.96% | -6.96% |
| fee 0 | -6.62% | -6.62% |
| fee +1bp | -6.66% | -6.66% |
| fee +2bp | -6.65% | -6.65% |
| latency x2 | -6.97% | -6.97% |
| latency x4 | -6.98% | -6.98% |
| queue risk-averse | -6.94% | -6.94% |
| queue log | -6.95% | -6.95% |
| size x2 | -6.84% | -6.84% |
| floor x0.8 | -7.02% | -7.02% |
| floor x1.25 | -6.95% | -6.95% |
| kappa x0.5 | -6.96% | -6.96% |
| kappa x2 | -6.96% | -6.96% |

Its return depends on one thing: whether the 8 bp spread floor is under one
price tick. Under one tick it quotes at the touch, fills about 77,000 times a
session with almost no drawdown and earns the 0.5 bp rebate on ~5,000x
capital turned over (+19% a session); at one tick or more it loses 25 bp,
flattens and stays paused (median 0.92 of the session), every session.

| block | sessions | floor under 1 tick | median return, floor < 1 tick | median return, floor ≥ 1 tick | median fills, floor < 1 tick | median fills, floor ≥ 1 tick |
| --- | --- | --- | --- | --- | --- | --- |
| training | 48 | 9 | +19.23% | -0.25% | 78,825 | 792 |
| validation | 27 | 26 | +19.61% | +11.14% | 76,183 | 44,852 |
| test | 27 | 0 | – | -0.25% | – | 415 |

The floor was under one tick on 9 training and 26 validation sessions (late
January to June 2026) and on no test session. So the +11,024% validation
return is the rebate on a sub-tick quote, and the −6.96% test is 27 loss-limit
trips; the fee stresses barely move it because it has stopped trading. Neither
says anything about market making on ENAUSDT. It is reported and left out of
the sums. (Figure: `docs/assets/c13/ena.png`.)

### BNBUSDT: validation and test

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (momentum, markout half-life) | +46.15% | +1.11% | **-0.63%** | -1.43% | -1.27% | 12/27 | 79 |
| pick, crash-guarded: family: breaker [crash-guarded] | +3.70% | +0.77% | **+1.06%** | +0.49% | +0.74% | 17/27 | 38 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -0.63% | +1.06% |
| fee 0 | -1.27% | +0.74% |
| fee +1bp | -2.54% | +0.11% |
| fee +2bp | -3.80% | -0.51% |
| latency x2 | -0.63% | +1.17% |
| latency x4 | -0.52% | +1.39% |
| queue risk-averse | -0.80% | +1.06% |
| queue log | -0.31% | +1.65% |
| size x2 | -0.95% | +1.89% |
| floor x0.8 | -0.94% | +1.49% |
| floor x1.25 | -1.12% | +0.08% |

**BNBUSDT verdict.** The crash-guarded pick (GLFT 4 bp + breaker) makes
+1.06% on the test, +0.49% without its best session and +0.74% without the
rebate, positive in 9 of 10 stresses, 38 fills a session; it loses at +2 bp
(−0.51%) and is +0.11% at +1 bp. C8's rule picked the crash winner
(+42.2% on the crash session, +2.8% on the other 47), which lost under every
stress. There is no C8 choice for BNB to compare with.

### DOGEUSDT: validation and test

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (momentum, breaker) | +132.56% | +0.38% | **-5.01%** | -5.19% | -5.22% | 9/27 | 21 |
| pick, crash-guarded: greedy [crash-guarded] (loss limit, inventory flatten) | -1.64% | +1.13% | **-2.32%** | -2.75% | -2.51% | 13/27 | 18 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -5.01% | -2.32% |
| fee 0 | -5.22% | -2.51% |
| fee +1bp | -5.64% | -3.04% |
| fee +2bp | -6.06% | -3.39% |
| latency x2 | -2.84% | -0.88% |
| latency x4 | -4.12% | -3.04% |
| queue risk-averse | -4.73% | -1.31% |
| queue log | -3.51% | -1.60% |
| size x2 | -9.50% | -4.33% |
| floor x0.8 | -3.75% | -2.65% |
| floor x1.25 | -5.24% | -1.81% |
| kappa x0.5 | -3.91% | -2.04% |
| kappa x2 | -4.40% | -1.85% |

**DOGEUSDT verdict.** Both picks lose on the test and under every stress
(crash-guarded −0.9% to −4.3%, C8's rule −2.8% to −9.5%). No edge.

### LINKUSDT: validation and test

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (momentum, breaker, markout half-life) | +54.22% | -3.67% | **-0.86%** | -1.79% | -2.29% | 14/27 | 215 |
| pick, crash-guarded: family: time of day | +13.02% | +0.19% | **-0.45%** | -1.44% | -0.78% | 18/27 | 40 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -0.86% | -0.45% |
| fee 0 | -2.29% | -0.78% |
| fee +1bp | -5.07% | -1.45% |
| fee +2bp | -7.78% | -2.11% |
| latency x2 | -3.31% | +2.58% |
| latency x4 | -4.48% | +4.17% |
| queue risk-averse | -1.17% | +0.60% |
| queue log | -0.75% | +0.35% |
| size x2 | -0.78% | +4.18% |
| floor x0.8 | -1.04% | +2.23% |
| floor x1.25 | -1.46% | +1.05% |
| kappa x0.5 | -3.32% | +2.33% |
| kappa x2 | -1.69% | -1.39% |

**LINKUSDT verdict.** Both picks lose on the test (C8's rule −0.86%,
crash-guarded −0.45%, a time-of-day filter on the grid quoter) and at every
fee level; the crash-guarded pick's stresses scatter from −2.11% to +4.18%
(latency ×4 +4.17%, size ×2 +4.18%, κ ×2 −1.39%), which is what no edge
looks like.

### ADAUSDT: validation and test

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (momentum, breaker, time of day, stop) | +58.54% | -0.84% | **-9.10%** | -9.77% | -11.31% | 5/27 | 354 |
| pick, crash-guarded: family: momentum [crash-guarded] | +18.93% | +4.20% | **-2.20%** | -3.00% | -3.01% | 14/27 | 115 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -9.10% | -2.20% |
| fee 0 | -10.93% | -3.01% |
| fee +1bp | -14.48% | -4.61% |
| fee +2bp | -17.75% | -6.19% |
| latency x2 | -8.56% | -1.24% |
| latency x4 | -7.93% | -1.32% |
| queue risk-averse | -9.11% | -3.92% |
| queue log | -9.08% | -2.73% |
| size x2 | -12.50% | +1.67% |
| floor x0.8 | -9.19% | -1.97% |
| floor x1.25 | -8.64% | -2.34% |

**ADAUSDT verdict.** Both picks lose on the test (C8's rule −9.10%,
crash-guarded −2.20%) and under every stress but one (crash-guarded at
double size, +1.67%). No edge.

### LTCUSDT: validation and test

| entry | training | validation | test | test without its best session | test without the rebate | test sessions up | fills/session (test) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pick, C8's rule (all sessions): greedy (momentum, toxicity sizing, time of day, signal smoothing) | +48.39% | -2.59% | **-5.06%** | -5.34% | -6.04% | 3/27 | 347 |
| pick, crash-guarded: family: breaker [crash-guarded] | +10.42% | -2.58% | **+9.66%** | +8.07% | +8.63% | 18/27 | 111 |

| stress (test block) | pick, C8's rule | pick, crash-guarded |
| --- | --- | --- |
| as chosen | -5.06% | +9.66% |
| fee 0 | -6.04% | +8.63% |
| fee +1bp | -7.98% | +6.61% |
| fee +2bp | -9.88% | +4.62% |
| latency x2 | -4.90% | +9.51% |
| latency x4 | -4.86% | +9.64% |
| queue risk-averse | -5.27% | +9.13% |
| queue log | -5.52% | +9.63% |
| size x2 | -8.09% | +11.61% |
| floor x0.8 | -4.66% | +10.67% |
| floor x1.25 | -5.11% | +9.57% |

**LTCUSDT verdict.** The strongest result of the contest. The crash-guarded
pick (hfgrid 4 bp + a slow volatility breaker: 3000 s cooldown, vol 4) makes
**+9.66%** on the test, +8.07% without its best session and +8.63% without
the rebate, positive under every stress including Bybit's standard +2 bp
fee (+4.62%); 18 of 27 sessions up, best session +1.47%, worst session
drawdown 1.44%, about 110 fills a session. Its crash session: −1.35%,
drawdown 5.6%. Caveat: on validation alone it lost (−2.58%, 13 of 27 up);
it was picked on training + validation combined (+10.42% on training), so
validation by itself would not have chosen it. C8's rule picked the crash
winner (+61% on the crash session), which lost −5.06% (3 of 27 sessions
up).

### Scoreboard (locked test, 27 sessions Jun-Oct 2026, the five original contracts)

| contract | pick, all sessions | pick, crash-guarded | C8's choice, unchanged | best contest pick at maker fee +1 bp |
| --- | --- | --- | --- | --- |
| SUIPERP | −1.01% | +0.22% | +1.68% | crash-guarded −0.29% |
| AVAXUSDT | −1.26% | −1.69% | +2.64% | crash-guarded −3.76% |
| XRPPERP | −4.54% | +1.22% | +1.37% | crash-guarded +0.14% |
| XRPUSDT | −0.18% | −1.01% | −0.40% | all sessions −3.81% |
| SUIUSDT | −4.81% | **+4.60%** | −4.64% / −4.99% | crash-guarded **+0.84%** |
| sum of the five | −11.80 | **+3.34** | +0.65 (with SUIUSDT OBI; +0.30 with SUIUSDC OBI) | |

C8's rule (rank by chained return over all training sessions) picked the
crash-session winner every time and lost on all five tests (−11.8 points in
total). The crash-guarded rule is ahead of it on three of five (on XRPUSDT and AVAXUSDT both lose) and, summed
over the five contracts, ahead of C8's untouched Binance choices (+3.34 vs
+0.65 points). It produced one clear winner (SUIUSDT, a grid quoter with a
volatility breaker) and one break-even (XRPPERP). At Bybit's standard maker
fee (+2 bp) every pick loses; at +1 bp two stay positive (SUIUSDT +0.84%,
XRPPERP +0.14%).

### Scoreboard, all tested contracts (from `report.py`; fee columns are the backtest's own fee stresses)

| contract | pick, C8's rule | pick, crash-guarded | C8's choice, unchanged | crash-guarded at maker fee +1 bp | crash-guarded at +2 bp |
| --- | --- | --- | --- | --- | --- |
| XRPPERP | -4.54% | +1.22% | +1.37% | +0.14% | -0.57% |
| SUIPERP | -1.01% | +0.22% | +1.68% | -0.29% | -0.63% |
| XRPUSDT | -0.18% | -1.01% | -0.40% | -4.66% | -7.01% |
| SUIUSDT | -4.81% | +4.60% | -4.64% | +0.84% | -1.60% |
| AVAXUSDT | -1.26% | -1.69% | +2.64% | -3.76% | -5.12% |
| ENAUSDT (artefact) | -6.96% | -6.96% | – | -6.66% | -6.65% |
| BNBUSDT | -0.63% | +1.06% | – | +0.11% | -0.51% |
| DOGEUSDT | -5.01% | -2.32% | – | -3.04% | -3.39% |
| ADAUSDT | -9.10% | -2.20% | – | -4.61% | -6.19% |
| LINKUSDT | -0.86% | -0.45% | – | -1.45% | -2.11% |
| LTCUSDT | -5.06% | +9.66% | – | +6.61% | +4.62% |
| sum, five original | -11.80 | +3.33 | +0.64 | | |
| sum, added | -20.65 | +5.75 | – | | |

Over the five added contracts with a usable result (ENAUSDT excluded), C8's
rule loses −20.65 points and the crash-guarded rule makes +5.75: it is ahead
on all five, with two picks that hold up (LTCUSDT +9.66%, BNBUSDT +1.06%)
and three that lose (DOGEUSDT, ADAUSDT, LINKUSDT). Over all ten: C8's rule
−32.45 points, lost on every contract; crash-guarded +9.08, ahead on eight
of ten.

Every crash-guarded final pick passes the guard (crash-session drawdown
0.7-9.3%); the gap noted under the crash-session check affects only stage-3
results that added nothing to the screen leader, and in each such case a
family variant that passes the guard was picked instead.

## Status

See `python experiments/c13_bybit_contest/contest.py status`. Kaggle runs use
`units` dumps and 3 workers per notebook (4 workers ran out of memory on
XRPUSDT sessions).
