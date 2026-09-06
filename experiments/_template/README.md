# Experiment template

Copy this directory to `experiments/<phase-id>_<slug>/` when starting a phase.
It exists to show the conventions in `docs/experiment-protocol.md` working end to
end — it deliberately does **not** implement any ladder item.

```bash
python experiments/_template/run_example.py --seeds 5 --steps 500
```

That writes a run directory under `runs/template/`, logs metrics as JSONL, and
saves a mean±std plot. Read the ~60 lines of `run_example.py` before A1; every
later phase follows the same shape.
