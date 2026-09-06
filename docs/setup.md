# Setup

Python 3.11. [uv](https://docs.astral.sh/uv/) for environments — it is what the
Makefile assumes, but plain `pip` works identically if you prefer.

## Base environment

```bash
git clone https://github.com/davidmiheev/rl-ladder && cd rl-ladder
uv venv --python 3.11 && source .venv/bin/activate
make install          # core deps + pytest/ruff, no torch
make test             # should pass in a couple of seconds
```

`make install` deliberately stays light: numpy, matplotlib, tqdm, pyyaml. The
heavy stacks are per-part extras, installed only when you reach that part.

## Per-part extras

```bash
make install-a        # torch, gymnasium[classic-control,box2d], SB3, tensorboard
make install-b        # transformers, TRL, PEFT, datasets, accelerate, bitsandbytes
make install-c        # hftbacktest, numba, pandas, polars, pyarrow
```

Or with uv directly: `uv pip install -e ".[part-a,dev]"`.

### Part A notes

- **Box2D** (LunarLander, BipedalWalker) needs a compiler and SWIG. On Debian/Ubuntu:
  `sudo apt-get install -y build-essential swig`.
- **CPU torch** is the sensible default for Part A and saves ~2 GB of CUDA wheels:
  `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
  *before* `make install-a`, so the resolver keeps it.
- **Atari** (the optional A3 item) and **MuJoCo**/**PettingZoo** (A5 optional, A7)
  live in the `part-a-extra` group; Atari ROMs come with `ale-py>=0.10`, no
  separate ROM download step.

### Part B notes

Part B needs a GPU. See [`compute.md`](compute.md) for pod sizing. On a fresh pod:

```bash
uv pip install -e ".[part-b,dev]"
huggingface-cli login                       # gated models + pushing checkpoints
export HF_HOME=/workspace/.cache/huggingface   # keep the cache on the big volume
```

`part-b-extra` (vLLM, flash-attn) is optional and CUDA-specific — add it only
once the basic TRL loop runs, and expect flash-attn to compile for several
minutes.

### Part C notes

`hftbacktest` ships prebuilt wheels for the Python binding, so `make install-c`
needs no Rust toolchain. You only need Rust (`rustup`) if you end up modifying
the backtester itself. Market data is a separate step — see [`data.md`](data.md).

## Companion repositories

```bash
make companions       # clones the 7 lab/reference repos into ./companions
```

See [`companion-repos.md`](companion-repos.md) for the pins and what each is for.

Running Practical_RL / nlp_course notebooks needs their own dependencies, which
conflict with this project's pins in places (older gym, older transformers).
Give them a separate venv rather than fighting the resolver:

```bash
uv venv --python 3.11 .venv-labs && source .venv-labs/bin/activate
uv pip install -r companions/Practical_RL/... # per-week requirements
```

Both courses also publish Colab links per week — for the labs specifically, Colab
is often the lower-friction path, and nothing in the labs needs to share an
environment with this repo's code.

## Verifying the install

```bash
make test                                   # plumbing smoke tests
python experiments/_template/run_example.py # writes a run dir + a plot
make lint
```

The template script is a plumbing demo (a random-policy bandit baseline), not an
A1 deliverable — see [`experiment-protocol.md`](experiment-protocol.md).
