# Experiment protocol

Rules that keep 59 checklist items from turning into 59 unreproducible
screenshots.

## What "done" means

A checklist item in [`ladder.md`](ladder.md) may be ticked when:

1. Code for it exists under `experiments/<phase>/`.
2. It has been run over **at least 3 seeds** (5 preferred for anything noisy).
3. The phase README has a results row: config, metric, value, run directory.
4. The README's **Findings** section says what actually happened — including when
   the answer is "no difference" or "it did not reproduce".

Several items are explicitly designed to fail (the near-zero KL penalty in B1,
the DPO overfit in B2, the gameable verifier in B3). For those, the failure *is*
the deliverable; record it precisely enough that you could reproduce it.

## Runs

Use `rl_ladder.Run`; do not invent per-phase logging.

```python
from rl_ladder import Run, seed_everything

run = Run(phase="a3", name="dqn-no-target-net", config=vars(args))
seed_everything(args.seed)
...
run.log(step=step, episode_return=ret, loss=loss)
run.finish({"mean_return_last_100": float(np.mean(returns[-100:]))})
```

That writes `runs/<phase>/<name>__<timestamp>/` containing `config.json`,
`env.json` (git sha, dirty flag, python, platform, argv), `metrics.jsonl`, and
`summary.json`.

**`runs/` is gitignored.** Run artifacts are disposable; the committed record is
the phase README row plus, when a plot earns it, a copy in `docs/assets/`. The
git sha in `env.json` is what makes a stale run traceable — commit before a long
run so `git_dirty` is `false`.

## Seeds

`seed_everything(seed)` seeds `random`, numpy, and torch when installed. Sweep
seeds explicitly (`for seed in range(5)`), never a random seed. A single-seed RL
curve is not evidence — variance between seeds routinely exceeds the effect being
measured, which is exactly the trap several ablations here are meant to expose.

## Plots

`plot_curves` takes one list of per-seed curves per label and draws mean ±1 std.
Every curve in this repo is a multi-seed band; if you find yourself plotting a
single run, that is a signal you have not run enough seeds yet.

```python
plot_curves(
    {"DQN": dqn_curves, "Double DQN": ddqn_curves},
    title="LunarLander return", xlabel="episode", ylabel="return",
    out_path=run.path("returns.png"), smooth_window=20,
)
```

## Ablations

Most items are of the form "remove X, show what breaks". Change **one** thing per
run, keep the seed set identical across the arms of a comparison, and give each
arm a run name that says what it is (`dqn-baseline` / `dqn-no-replay` /
`dqn-no-target-net`). Comparing arms trained on different seed sets is the
easiest way to draw a confident wrong conclusion.

## Cross-checking against reference implementations

Several phases ask you to compare your implementation against TRL or SB3. When
they disagree, assume yours is wrong until proven otherwise, and diff on the
specifics: advantage normalization, whether the loss is summed or averaged over
tokens, discounting at episode boundaries, and where the KL term enters (reward
vs. loss). Record the discrepancy in the phase README even after fixing it —
those are the most useful notes in the whole repo.

## Comparing against a baseline (Part C especially)

Part C's whole point is beating a calibrated fixed formula. That comparison is
only meaningful on **held-out days that were never used for calibration or
training**. Decide the train/held-out split once, in C1, write it into the C1
README, and reuse it unchanged for C3–C5. Re-picking the split after seeing
results is how a backtest lies to you.

## Items are written to be self-contained

`ladder.md` is the teaching document, not an index. Each phase states the
problem it solves and where it sits in the sequence; each item explains what is
being asked, how to approach it, and what to expect — enough that no paper or
notebook is needed to understand the task. Links are for depth, not for
comprehension.

Two constraints keep that honest:

- **Checkbox lines stay verbatim** from the original tracker, so a tick always
  refers to the item as written. Explanations go in an indented paragraph
  beneath. The correspondence is checked mechanically against the archived HTML.
- **Explanations state what to expect, including the failure.** Several items
  are designed to fail; saying so up front is the difference between a reader
  reproducing the lesson and thinking their code is broken.

When a phase is finished, fold what the runs actually taught back into its item
text if the original explanation turned out to be misleading — the A3 items now
warn about measuring overestimation too early, because that is what the runs
showed.

## Writeups cite their sources

Each phase gets `docs/writeups/<phase>.md` explaining what the phase asks, why,
and what the results mean. It ends with a **"Where these items come from"**
table mapping every checklist item to the source it was drawn from — book
section, original paper, or lab notebook — with links, plus a sentence on why
that item earned a place. The point is that the curriculum should be auditable:
a reader can check whether an item is a faithful reproduction of a known
experiment or someone's invention.

Two rules learned the hard way:

- **Verify numbering before citing it.** Section, example and figure numbers
  differ between the free draft of Sutton & Barto and the published 2nd edition,
  and a citation that sends a reader to the wrong figure is worse than none. The
  Dyna maze is figure 8.3; figure 8.2 is the architecture diagram — this repo
  cited 8.2 in two places before anyone checked.
- **Separate sources from inferences.** Where a writeup draws its own connection
  between phases, it says so rather than implying the source made the claim.

Formulae in writeups are LaTeX (`$...$` inline, `$$...$$` display), which GitHub
renders natively.

## Directory conventions

```
experiments/<phase-id>_<slug>/     one directory per ladder phase
  README.md                        goal, how to run, results table, findings
  <script>.py                      the implementation
runs/<phase>/<name>__<timestamp>/  generated, gitignored
docs/assets/                       plots worth committing
```

Phases are self-contained on purpose: A3's DQN is meant to be readable top to
bottom without chasing helpers. Shared code in `src/rl_ladder/` stays limited to
seeding, run logging, and plotting. Reuse *across* phases happens by copying and
adapting (A4's REINFORCE into B1, A5's TD3 into C3) — the ladder asks you to
re-derive things, and copying makes the diffs visible.
