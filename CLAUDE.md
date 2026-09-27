# Working rules for this repository

## Writeups

- Every writeup under `docs/writeups/` must carry its phase's **key tables and
  figures**, not only the lab notebook (`experiments/*/README.md`). A number
  that decides a conclusion appears in a table in the writeup; a comparison
  that decides a conclusion appears as a figure. Figures live in
  `docs/assets/<phase>/` and are produced by a script in the phase's
  experiment directory, so they can be regenerated from `runs/`.
- When a new result lands, update the writeup's tables, figures, at-a-glance
  box and contents list in the same commit, then rebuild the PDF
  (`cd docs/pdf && node build.mjs <name>.md`) and commit both.
- Before finishing a phase, check that every figure in `docs/assets/<phase>/`
  is referenced by a writeup and every results row in the README has a place
  in the writeup.

## Part C (market making): no untested verdicts

- Do not reject, rank or estimate the value of an idea without running it
  through the full backtest and a stress test. Priors ("probably won't help",
  "not expected to change the conclusion") are not results; write them down
  as hypotheses and test them.
- Every configuration reported as a result is chosen on training sessions
  only, reported on held-out sessions, and stress-tested: maker fee without
  the rebate, higher order latency, a more pessimistic queue model, the
  neighbourhood of the chosen parameters, and with the best and the crash
  session (2025-09-22) removed.
- Report fills alongside PnL, so a configuration that stops trading never
  passes for one that makes money.

## Secrets

- Tokens (Hugging Face, Kaggle) are passed as environment variables only and
  are never written to the repository, configs or logs.
