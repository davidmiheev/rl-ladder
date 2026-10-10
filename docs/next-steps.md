# Next steps after C13 — strategies, fixes and readability

Handover list for the next session. Written 2026-10-10 at the end of C13
(branch `claude/rl-ladder-setup-docs-d43b31`,
[PR #1](https://github.com/davidmiheev/rl-ladder/pull/1)). Items are ordered by
priority inside each section; each says why, what, where, what "done" means
and roughly what it costs. The repository rules in `CLAUDE.md` apply to all of
it: no untested verdicts, every reported configuration chosen on training,
reported held out and stress-tested, fills next to PnL, and writeup tables,
figures and PDF updated in the same commit as the result.

## Where things stand (read first)

| what | result | where |
| --- | --- | --- |
| C8's choices (Binance) on a year of Bybit | the edge does not travel; one of six makes money without the crash | `docs/writeups/c10-forward.md` |
| C8's selection rule on Bybit (C13) | picks the 2025-10-10 crash winner; loses on all 10 contracts with a usable result, −32.45 points on the locked test | `docs/writeups/c13-bybit-contest.md` §4-§5 |
| C10's crash-guarded rule (C13) | +9.08 points over the 10; ahead of C8's rule on 8 of 10 | same, §5 |
| What holds up | a slow quoter with a volatility breaker: **LTCUSDT +9.66%** (hfgrid 4 bp, breaker cooldown 3000 s, vol 4; positive under every stress, +4.62% at +2 bp), **SUIUSDT +4.60%** (hfgrid 4 bp, breaker cooldown 300 s, vol 2.5; fails at +2 bp), **BNBUSDT +1.06%** (GLFT 4 bp + breaker; fails at +2 bp) | same, §5-§6 |
| Known backtest artefacts | positions overshoot the cap when a sweep fills several resting orders (up to 2.6× capital on the crash); a spread floor under one tick farms the 0.5 bp rebate (ENAUSDT, +11,024% on validation) | same, §4 and §5 |

The locked test block (2026-06-18 .. 2026-10-02, every 4th day) has now been
used. **Nothing may be chosen on it any more.** Any new choice needs new
held-out sessions (§B1).

## A. Protect what exists (do first; cheap, and losing it is expensive)

**A1. Back up the result cache — done 2026-10-10.** `runs/c8/days/` (≈391 MB
on disk, 23 MB compressed, 2,434 configuration-session keys) holds every
backtest of C8, C10, C12 and C13; `runs/` is gitignored. A copy is in the
private Kaggle dataset `davidmiheev/rl-ladder-c8-days`, checked by
downloading it back and comparing byte for byte; the restore command is in
`docs/data.md`. **Upload a new version after every batch of new runs**
(command in the same place), or the backup goes stale.

**A2. Back up the C11 L3 recording — done 2026-10-10.** The 385 MB
Bitstamp/Bitfinex L3 recording (65 hourly files, 2026-10-08 17:00 to
10-09 05:00 UTC) is in the private Kaggle dataset
`davidmiheev/rl-ladder-c11-l3`, checked by a byte-identical restore; the
restore command is in `docs/data.md`. C11 itself (converter and queue-model
calibration, `experiments/c11_queue_l3/`) is unfinished and has no README
yet.

**A3. Kaggle tooling is now in `scripts/kaggle/`** (`gen_c13.py` builds a
self-contained notebook with the code embedded; `kpost.py` saves and runs it
via Kaggle's MCP endpoint; `kcancel.py` cancels a session). The token comes
from `KAGGLE_API_TOKEN` only. Still missing: a `scripts/kaggle/README.md`
with the workflow in §E below, and a generator for C10/C12 runs (the C13 one
is the only one committed).

## B. Strategy: confirm and extend what works

**B1. Forward-test the three survivors on new sessions.** LTCUSDT, SUIUSDT
and BNBUSDT crash-guarded picks, unchanged, on Bybit sessions after
2026-10-02 as they become available (download with
`experiments/c10_forward/bybit.py`), with C8's stress set. This is the only
untouched held-out data left. Also run them on **every day** of the test
block, not every 4th: no choice is made, so it adds statistical power
without reusing the block for selection (report it as "test block, all
days", separately from the locked-test numbers). Done when each pick has a
forward table (return, ex-best, no rebate, +1/+2 bp, fills, sessions up) in
the C13 README and writeup. Cost: the block has 107 days of which 27 ran,
so ≈80 extra days × 3 contracts × ~12 configurations ≈ 2,900 runs (one
round of five notebooks); forward
sessions as they arrive.

**B2. Treat "slow quoter + volatility breaker" as a hypothesis and test it
as one rule.** All three survivors share it. Fix one recipe in advance
(hfgrid 4 bp, breaker cb_mode 1, cb_fast_hl 100, cooldown and vol chosen on
training only) and run it on all 11 contracts, on new sessions (B1's
dates). Done when the table says on how many contracts the fixed rule holds
up, with stresses. Do not write "the pattern generalises" before that table
exists.

**B3. Choose with the fee in the objective.** Every backtest assumes
Binance's −0.5 bp maker rebate; Bybit's standard maker fee is +2 bp, and at
+2 bp only LTCUSDT survives. Rerun stage 1-4 selection with the objective
computed at maker fee 0 and at +1 bp (the backtest's own fee setting, not a
turnover recomputation; ENAUSDT shows why), on training + validation only.
First find out which fee tier is actually available (Bybit VIP / market
maker programme) and use that as the headline fee. Cost: the screen and
families stages again per fee level (≈ the original C13 budget each).

**B4. Reject sub-tick spread floors in the screen.** Add a filter to
`contest.py` (and `c8.py`'s grid) that drops any setting whose median
`spread_floor_tick` on the screen sessions is below 1 (or 2) ticks, and
report what it removes. ENAUSDT's leader and pick are the case to check
against. Cheap: the per-session records already carry `spread_floor_tick`.

**B5. Fix the position-cap overshoot in the backtest.** The cap is checked
when quotes are placed and ignores resting orders, so a sweep through
several levels overshoots it (2.19× on XRPPERP, 2.63× on AVAXUSDT on the
crash session; C8's SUIUSDT choice loses more than 100% that day). Make the
quoter count resting orders against the cap (worst-case inventory if all
fill). Then rerun the crash session for every final pick and C8 choice and
report the change. This changes backtest semantics: do it on a branch of
`c6_strategy_zoo/strategies.py`, rerun the affected sessions, and state in
each writeup which numbers predate the fix.

**B6. Close the crash guard's gap.** Stage 3 starts from the screen leader
and the guard only applies to what is added. It did not change any final
pick in C13 (all crash-guarded picks pass, 0.7-9.3% crash drawdown), but it
should be fixed before reuse: apply the guard to the base too
(`contest.greedy`), and if the leader fails, start from the best screen
setting that passes.

**B7. Better selection statistics.** The chained return over 48 sessions is
dominated by a few sessions. Try, on training + validation only:
median session return, return without the best session, and a deflated
Sharpe ratio that accounts for the ≈54 × families × greedy configurations
tried. Compare which picks each would have made; evaluate on B1's new
sessions, not on the used test block.

**B8. A validation veto, as a hypothesis.** LTCUSDT's pick lost 2.58% on
validation and then made +9.66% on the test; requiring a positive validation
return would have dropped it. Decide the rule (veto or not) before seeing
B1's results, and test it there.

**B9. A small portfolio.** Run the three survivors together (equal capital)
on the test block and the new sessions; report the portfolio's return,
drawdown and the correlation of their session returns. Sums of per-contract
returns ("points") in the writeup are not a portfolio.

**B10. Carry-overs from the Part C overview §8.** Record Binance live from
an always-on Tokyo host to separate "venue" from "time"; measure the real
arrival delay before more lead-lag work; C11 L3 queue calibration (the
queue model decides whether quoting at the touch fills, which is what made
ENAUSDT's artefact possible); OKX as a third venue; RL with a privileged
critic only once forecasts exceed R² 0.1.

**B11. Optional: finish C12's year.** C12 ran only its 7-session held-out
week; its `test`, `train-long` and `test-long` phases (≈610 units) never
ran, because C13 superseded it. Run them only if the dashboard's C12 line
over the year is wanted; record the reason in `experiments/c12_bybit_select/README.md`
either way.

## C. Readability of results

**C1. Dashboard: show the test block, not the in-sample run-up.** In the
contest cards each entry is one compounded line over training + validation
+ test, so the crash spike sets the axis and the test is a flat stub at the
right edge. Changes in `experiments/c13_bybit_contest/dashboard/index.html`:
shade the three blocks; add a second chart per contract rebased to zero at
the first test session (like `docs/assets/c13/test.png`); give the final
picks their own colours (they share them with the stage-3 lines); give
ENAUSDT its own scale or an "artefact" badge; stop compounding below −100%
(C8's SUIUSDT choice), plotting session sums or clipping with a note
instead.

**C2. Dashboard: fee columns and a scoreboard.** Add the backtest's own
+1 bp and +2 bp stress results to the contest dataset (`build.py`) and to
the leaderboard, and a scoreboard card that mirrors the writeup's §5 table.
The leaderboard's "without the rebate" column is a turnover recomputation
and is wrong for ENAUSDT; label it as such or replace it with the backtested
fee-0 stress.

**C3. Dashboard: rebuild the sessions dataset from a script.** The
`sessions` dataset (C10/C12 lines) was assembled by hand and committed as
`dashboard/sessions.json`; write `dashboard/build_sessions.py` that
regenerates it from `runs/`.

**C4. Writeup: a scoreboard figure.** The C13 decision (C8's rule vs
crash-guarded vs C8 unchanged, at fee −0.5/0/+1/+2 bp) is a table only.
Add a bar or dot figure to `report.py` (`scoreboard.png`) and put it at the
top of §5; `CLAUDE.md` asks for a figure where a comparison decides a
conclusion.

**C5. Writeup: a stress heatmap instead of (or next to) the dot strip.**
Contract × stress, coloured by test return, one panel per rule; easier to
read than `stress.png` and than the two stress tables, which could then move
to an appendix.

**C6. Writeup: shorten §5.** The per-contract tables (10 of them) make §5
long. Keep the scoreboard and the three survivors in the body; move the rest
to an appendix generated by `report.py --tables`.

**C7. Say "points" less.** Sums of per-contract returns are reported as
"points"; readers take them for a portfolio return. Replace with the
portfolio of B9 once it exists, or with "contracts up / down" plus the
median.

**C8. Lab notebook hygiene.** `experiments/c13_bybit_contest/README.md`
carries interim snapshots and two later corrections (stress counts, the
guard gap). Fold the corrections into the text and mark interim tables as
superseded, so a reader does not have to read it in order.

**C9. One-page results index.** `docs/writeups/c0-part-c-overview.md` §3 is
the place for "every method, held out"; add C13's three survivors there with
the same columns as the other rows, and keep the overview's PDF rebuilt.

## D. Small fixes found along the way

- `contest.py run --phase` accepts only the five stage names, also when a
  units file decides the work; a pooled batch labelled "mix" did nothing.
  Either accept any label when `C13_UNITS` is set, or document it (§E).
- `report.py` takes several minutes because it reruns `final_pick`, which
  ranks every candidate; cache the picks per run.
- `c8.results` returns `None` when one session is missing, which silently
  drops an entry from tables and figures; log which entry and session.
- The C13 README's §"Stages 2-3, training" table is a round-3 snapshot;
  `report.py`'s crash table is the final one.

## E. How C13-style runs were done (for the next session)

1. Dump pending work: `C13_CONTRACTS=A,B,... python experiments/c13_bybit_contest/contest.py units --phase <screen|families|greedy|valid|test> > units.json`.
   Later phases depend on earlier results being in `runs/c8/days`.
2. Several phases can share one batch: concatenate their units files; the
   run command then ignores the phase except as a label, but the label must
   still be one of the five names.
3. Build notebooks: `python scripts/kaggle/gen_c13.py 0,1,2 15 test units.json > nb-0.py`
   (three shards per notebook; two when XRPUSDT sessions are in the batch,
   more run out of memory), and push with
   `python scripts/kaggle/kpost.py <slug> nb-0.py`. Five notebooks at a
   time is Kaggle's limit; a running session cannot be cancelled without
   its id. Each notebook stops itself at 10.75 h and checkpoints every
   10 minutes.
4. Collect: `kaggle kernels output <user>/<slug> -p <dir>`, then
   `cd runs/c8 && tar xzf <dir>/c10_days.tgz --skip-old-files`. Check each
   shard log for `SHARD DONE` and `FAILED` before relying on it.
5. Check with `contest.py status`, then `report.py` for tables and figures,
   then the writeup and `cd docs/pdf && node build.mjs <name>.md`.
6. Never print or write the Kaggle or Hugging Face tokens; redact `KGAT_`
   strings in any captured output.
