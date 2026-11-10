"""Build a self-contained Kaggle GPU notebook for Part B runs (code embedded as a base64 tarball).

    python scripts/kaggle/gen_b.py [--pip <requirement> ...] "<script> <args>" ["<script> <args>" ...] > notebook.py

Each quoted command is one run, e.g. "experiments/b1_ppo/reinforce_lm.py --seed 0". Runs are dealt
round-robin to the GPUs (a Kaggle T4 notebook has two) and each GPU works through its queue in order.
The run directories under runs/ are tarred to /kaggle/working/b_runs.tgz every 10 minutes and at the end.
No secrets: the models and datasets the runs use are public on the Hugging Face Hub.
"""
import base64, io, sys, tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FILES = ["src/rl_ladder/__init__.py", "src/rl_ladder/seeding.py", "src/rl_ladder/plotting.py",
         "src/rl_ladder/run.py", "experiments/a4_policy_gradient/pg.py",
         "experiments/b1_ppo/reinforce_lm.py", "experiments/b1_ppo/train_rm.py",
         "experiments/b1_ppo/ppo_lm.py", "experiments/b2_dpo/make_prefs.py", "experiments/b2_dpo/dpo_lm.py"]
argv, pip = sys.argv[1:], []
while argv and argv[0] == "--pip":
    pip.append(argv[1])
    argv = argv[2:]
commands = argv
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
    for f in FILES:
        tar.add(REPO / f, arcname=f)
code = base64.b64encode(buf.getvalue()).decode()
sys.stdout.write(f'''# rl-ladder Part B on a Kaggle GPU notebook: {len(commands)} run(s).
#
{"".join(f"#   python {c}{chr(10)}" for c in commands)}#
# The code is embedded below; models and data come from the public Hugging Face Hub.
import base64, io, os, subprocess, sys, tarfile, time
START = time.time()
REPO = "/tmp/rl-ladder"
OUT = "/kaggle/working"
CODE = "{code}"
COMMANDS = {commands!r}
def log(m):
    print(f"[{{(time.time() - START) / 60:6.1f}} min] {{m}}", flush=True)
os.makedirs(REPO, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz") as tar:
    tar.extractall(REPO)
# Kaggle's image ships torchao 0.10, which current PEFT refuses to import next to; nothing here uses it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "gymnasium", *{pip!r}], check=True)
import torch
n_gpu = max(torch.cuda.device_count(), 1)
log(f"ready, {{n_gpu}} GPU(s): {{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}")
# Mounted inputs (other notebooks' outputs): log the tree, and expose a reward model written by
# experiments/b1_ppo/train_rm.py (found by its rm_eval.json) to the commands as $RM_DIR.
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count(os.sep) <= 5:
        log(f"input: {{root}} {{sorted(files)[:5]}}")
    if "rm_eval.json" in files:
        os.environ["RM_DIR"] = root
    # A preference set written by experiments/b2_dpo/make_prefs.py, as $PREFS.
    if "prefs.jsonl.gz" in files:
        os.environ["PREFS"] = os.path.join(root, "prefs.jsonl.gz")
log(f"RM_DIR={{os.environ.get('RM_DIR')}} PREFS={{os.environ.get('PREFS')}}")
queues = [COMMANDS[i::n_gpu] for i in range(n_gpu)]
def worker(gpu, queue):
    script = "; ".join(f"python {{c}} > {{OUT}}/run_{{gpu}}_{{k}}.log 2>&1" for k, c in enumerate(queue))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    return subprocess.Popen(["bash", "-c", script], cwd=REPO, env=env)
procs = [worker(g, q) for g, q in enumerate(queues) if q]
def save():
    if os.path.isdir(f"{{REPO}}/runs"):
        with tarfile.open(f"{{OUT}}/b_runs.tgz.part", "w:gz") as tar:
            tar.add(f"{{REPO}}/runs", arcname="runs")
        os.replace(f"{{OUT}}/b_runs.tgz.part", f"{{OUT}}/b_runs.tgz")
last = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(30)
    if time.time() - last > 600:
        save()
        log("checkpoint")
        last = time.time()
save()
log(f"B-DONE exit codes {{[p.returncode for p in procs]}}")
''')
