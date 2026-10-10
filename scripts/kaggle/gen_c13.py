"""Build a self-contained Kaggle script running C13 contest shards (code embedded as a base64 tarball).

    python scripts/kaggle/gen_c13.py <shard,shard,...> <total-shards> <phase> [units.json] > notebook.py
"""
import base64, io, sys, tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# <shard,shard,...> <total>, e.g. "3,4,5,6 12"
shard_ids = [int(x) for x in sys.argv[1].split(",")]
total = int(sys.argv[2])
first, workers = shard_ids[0], len(shard_ids)
phase = sys.argv[3] if len(sys.argv) > 3 else "screen"
units_file = sys.argv[4] if len(sys.argv) > 4 else None
FILES = [
    "experiments/c1_glft_baseline/config.py", "experiments/c1_glft_baseline/glft.py",
    "experiments/c6_strategy_zoo/assets.py", "experiments/c6_strategy_zoo/assets.json",
    "experiments/c6_strategy_zoo/spread_floor.json", "experiments/c6_strategy_zoo/strategies.py",
    "experiments/c6_strategy_zoo/zoo.py", "experiments/c6_strategy_zoo/run_baseline.py",
    "experiments/c6_strategy_zoo/precompute.py", "experiments/c4_ppo_grpo/grid_search.py",
    "experiments/c4_ppo_grpo/fetch_train20.py", "experiments/c8_risk_controls/c8.py",
    "experiments/c10_forward/bybit.py", "experiments/c10_forward/forward.py",
    "experiments/c13_bybit_contest/contest.py",
    "experiments/c7_signals/snapshots.py",
    "src/rl_ladder/__init__.py", "src/rl_ladder/seeding.py", "src/rl_ladder/plotting.py",
    "src/rl_ladder/run.py",
]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
    for f in FILES:
        tar.add(REPO / f, arcname=f)
    if units_file:
        tar.add(units_file, arcname="runs/c13/units.json")
code = base64.b64encode(buf.getvalue()).decode()
shards = ", ".join(f"'{i}/{total}'" for i in shard_ids)
units_env = ' C13_UNITS="/tmp/rl-ladder/runs/c13/units.json",' if units_file else ""
sys.stdout.write(f'''# C13 (Bybit heuristic contest) on a Kaggle CPU session: shards {sys.argv[1]} of {total}.
#
# Runs rl-ladder's C13 (experiments/c13_bybit_contest/contest.py): C8's full
# selection procedure on a year of Bybit's free public order books and trades. No secrets are needed: the
# Bybit files are public. The per-session result cache is left in
# /kaggle/working/c10_days.tgz for the local machine to merge.
import base64, io, os, subprocess, sys, tarfile, time
START = time.time()
DEADLINE = START + 10.75 * 3600
REPO = "/tmp/rl-ladder"
OUT = "/kaggle/working"
CODE = "{code}"
def log(m):
    print(f"[{{(time.time() - START) / 60:6.1f}} min] {{m}}", flush=True)
os.makedirs(REPO, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz") as tar:
    tar.extractall(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hftbacktest==2.4.4", "orjson", "polars"], check=True)
log("ready")
env = dict(os.environ, C8_DEADLINE=str(DEADLINE), PYTHONUNBUFFERED="1", C13_PHASE="{phase}",{units_env}
           C13_CONTRACTS=os.environ.get("C13_SET", "XRPPERP,SUIPERP,XRPUSDT,SUIUSDT,AVAXUSDT"))
procs = [subprocess.Popen([sys.executable, "experiments/c13_bybit_contest/contest.py", "run",
                           "--phase", env["C13_PHASE"], "--shard", s],
                          cwd=REPO, env=env, stdout=open(f"{{OUT}}/c10_{{s.replace('/', '_')}}.log", "w"),
                          stderr=subprocess.STDOUT) for s in [{shards}]]
def save():
    if os.path.isdir(f"{{REPO}}/runs/c8/days"):
        with tarfile.open(f"{{OUT}}/c10_days.tgz.part", "w:gz") as tar:
            tar.add(f"{{REPO}}/runs/c8/days", arcname="days")
        os.replace(f"{{OUT}}/c10_days.tgz.part", f"{{OUT}}/c10_days.tgz")
while any(p.poll() is None for p in procs):
    time.sleep(600)
    save()
    log("checkpoint")
save()
log("C10-DONE")
''')
