# C6 — The strategy zoo: five published market makers, and RL on all five

*Status: running. Numbers below are filled in as each job lands; anything not
yet measured says so rather than carrying a placeholder.*

## Why this phase exists

C1 through C3 asked one question — can RL improve the GLFT coefficient set? —
and got a defensible but narrow answer: on ETHUSDC over five held-out
sessions, no learned configuration beat the fixed formula in absolute terms,
though every one of them lost less (best TD3 −0.63% against the baseline's
−8.44%).

The obvious objection is that GLFT was the wrong strategy to start from. It
loses to adverse selection on this venue, so "RL loses less than a losing
baseline" is a weak claim. Three things follow, and C6 does all three:

1. **Run the other four strategies hftbacktest publishes**, not just GLFT, so
   the result is about the method and not about one model.
2. **Give them a wider, asset-appropriate spread**, because C1's negative
   result was specifically that the calibrated width does not cover the
   adverse selection, and three of the five published strategies independently
   chose a half spread near 30 bp on the assets they were written for.
3. **Use more days**, on both sides of the split.

## What is held fixed across all five

The point of the phase is a comparison, so everything that is not the strategy
is identical: the same sessions, the same synthesised order latency, the same
`power2` queue model, the same Binance fee schedule (−0.005% maker,
0.07% taker), the same $5,000 orders against a $100,000 notional cap and a
$100,000 book, the same 100 ms reaction interval, and the same
`LinearAssetRecord(...).resample("10s").stats(book_size=100_000)` scoring call
C1 used.

Everything is parameterised in **basis points of price** rather than ticks.
The notebooks hard-code tick counts that only mean something for the asset
they were written for; `half_spread = 80` is 30 bp on 2023 BTC and nonsense on
XRP. Converting once, in `strategies.py`, is what makes "a wider spread,
appropriate to the asset" a single number and what lets the master table put a
spread column next to a spread column.

## The five

| name | notebook | fair price | half spread |
| --- | --- | --- | --- |
| `glft` | GLFT Market Making Model and Grid Trading | mid | closed form from a live fit of `A`, `k`, `σ` |
| `hfgrid` | High-Frequency Grid Trading | micro-price | `5 × σ` |
| `apt` | Market Making with Alpha — APT | `(1 + β·r_ref)·own_past` | 3 bp |
| `basis` | Market Making with Alpha — Basis | `ref + rolling basis` | 3 bp |
| `obi` | Market Making with Alpha — Order Book Imbalance | `mid + c₁·α` | 3.7 bp |

## The data, and the one substitution

The split is **widened, not re-picked**. `TRAIN_DAYS_EXT` and `TEST_DAYS_EXT`
in `experiments/c1_glft_baseline/config.py` are supersets of C1's five-day
lists on the same side of the same `SPLIT_DATE`, so nothing C1, C2 or C3
published moves. Training is ten of the twenty available days before
2025-09-15; held out is **every** day from 2025-09-15 to 2025-09-25, with no
selection at all.

APT and basis need a second instrument. The notebooks use spot against the
perpetual; this dataset has no spot feed. C6 trades the **USDC-quoted**
perpetual and uses the **USDT-quoted** one as the reference. Fitted on
training days the pair is

    β = 0.9913,  R² = 0.9985

and the median USDC/USDT basis over the eight pair days runs +0.56 to
+8.68 bp. So the pair is genuinely cointegrated and both strategies keep their
shape — but the economics change, and a weaker result is the expected one:

- the notebooks' basis is a perpetual's premium over spot, a funding and
  cash-and-carry object with a slow, sign-stable term structure;
- ours is a stablecoin spread plus a venue lead-lag: smaller, faster, and with
  no funding anchor pulling it back.

This is written down before the results because it is exactly the kind of
substitution that is easy to report as if it were the published experiment.

## The RL modification

One environment over all five (`env.py`). The action is four **multipliers**
that every strategy already has a slot for:

    spread_mult   on the half spread, whatever the strategy derives it from
    skew_mult     on the inventory skew
    alpha_mult    on the strategy's own fair-price signal
    grid_mult     on the grid step

All four are 1.0 at the published parameters, so each baseline is a strict
special case of its own policy — the property C2 built the GLFT environment
around, now holding five times over. `alpha_mult` is C5's item in disguise:
setting it to 0 turns a strategy into a pure mid-price quoter with the same
spread and skew, so what the policy chooses for it measures what that
strategy's signal is worth on these days.

TD3 only, two seeds. Both are compute decisions on a four-core machine, and
both are recorded as such in `train.py`: C3 already ran the three-way
algorithm comparison and TD3 won, and two seeds is enough to notice that a
result is seed-dependent but *not* enough to rank two configurations that are
close. C3 measured the seed spread on this problem at 3.74 points of return;
any C6 claim resting on a smaller gap is flagged rather than made.

## Checks that ran before any number was reported

`run_checks.py`. Two of them already earned their place by failing:

- **the running standardisation** that replaces the OBI notebook's
  O(window) `nanmean`/`nanstd` was dividing by window occupancy instead of by
  the count of finite entries, which moved the standardised imbalance by up to
  0.28 σ. Now equal to a brute-force recomputation to 2.8e-14.
- **the tape-based adverse-selection estimate** disagrees with the simulator
  by about 0.9 bp per fill, so it does *not* set the spread floor. See below.

The others: the skew unit conversion is exact to 5e-16 over 2,000 random
states; `baseline_action()` maps back to multipliers of 1.0; the reference
tables are strictly causal; the reward telescopes; an episode at the baseline
action reproduces the baseline runner; and `gymnasium.utils.env_checker`
passes.

## What a fill costs, and why the tape cannot set the spread

`measure_adverse_selection.py` replays ten training days and measures, for a
quote resting `d` ticks from the mid, the mid's drift over the following
holding horizon conditional on an aggressor reaching at least `d`. On ETHUSDC,
volume-weighted, at a 10 s horizon, the cost rises from 0.38 bp at the touch
to 2.2 bp at 200 ticks while the depth earned rises from 0.01 bp to 4.6 bp —
so on the tape a quote breaks even almost everywhere.

The simulator says otherwise: C1's `glft-grid` lost 0.80 bp per fill. The gap
is **fill selection**, which the tape cannot see. A resting quote is filled by
the subset of trades that consumed the queue ahead of it, and those are the
large informed sweeps. So the spread floor is set by `sweep_spread.py`, which
runs the strategy at each width in the simulator that does model queue
position and latency, on four training days only.

## Files

    zoo.py                        one njit quoting kernel, five fair-price models
    strategies.py                 published parameters, converted to bp of price
    precompute.py                 reference-leg tables for apt and basis
    env.py                        the Gymnasium environment
    train.py                      TD3 on the four multipliers
    evaluate.py                   held-out scoring, period and per-session
    run_baseline.py               fixed-parameter replays
    sweep_spread.py               half-spread and grid-depth sweeps
    measure_adverse_selection.py  the tape diagnostic
    master_table.py               the table
    plot_zoo.py                   equity and inventory figures
    run_checks.py                 the known-answer checks

## Reproducing

Everything is driven by one idempotent queue, because this machine kills long
jobs without warning and several hours of replays were lost before it
existed. `run_queue.py` enumerates every job the phase needs, asks `runs/`
which already have a `summary.json`, and runs the rest, so re-running after a
kill resumes rather than restarts.

```sh
python experiments/c6_strategy_zoo/run_checks.py          # known-answer checks first
python experiments/c6_strategy_zoo/assets.py              # measure and register the assets
python experiments/c6_strategy_zoo/run_queue.py --list    # what is left
python experiments/c6_strategy_zoo/run_queue.py --kind sweep      # spread floors
python experiments/c6_strategy_zoo/run_queue.py --kind baseline   # fixed parameters
python experiments/c6_strategy_zoo/run_queue.py --kind train      # TD3
python experiments/c6_strategy_zoo/run_queue.py --kind eval       # held-out scoring
python experiments/c6_strategy_zoo/master_table.py --compare
python experiments/c6_strategy_zoo/plot_zoo.py --days test --out docs/assets/c6
```

Shard across cores with `--shard i/n`. The queue refuses to start a widened
variant before its asset has a calibrated floor, so the sweeps can be run
first and a later pass picks up what they unblocked.

## Results

See `docs/writeups/c6-strategy-zoo.md`. The two tables are

```sh
python experiments/c6_strategy_zoo/master_table.py             # every row
python experiments/c6_strategy_zoo/master_table.py --compare   # baseline vs RL, paired
```

and the figures are `docs/assets/c6/{pnl,inventory}-<asset>.png`, with colour
carrying the strategy and line style the variant: dotted for the published
parameters, solid for the widened spread, dashed for TD3 on top of it.
