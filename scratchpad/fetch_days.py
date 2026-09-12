"""Fetch extra ETHUSDC days + the XRP USDC/USDT pair for the C6 strategy zoo."""
import sys, os
from pathlib import Path
from huggingface_hub import hf_hub_download

ETH_NEW = ["2025-08-29", "2025-09-03", "2025-09-05", "2025-09-10", "2025-09-12",
           "2025-09-16", "2025-09-17", "2025-09-19", "2025-09-21", "2025-09-23", "2025-09-24"]
XRP_DAYS = ["2025-08-26", "2025-08-29", "2025-09-03", "2025-09-08",
            "2025-09-15", "2025-09-18", "2025-09-22", "2025-09-25"]

jobs = []
for d in ETH_NEW:
    jobs.append((f"ethusdc_{d}.npz", f"ethusdc_{d}.npz"))
for d in XRP_DAYS:
    for sym in ("xrpusdt", "xrpusdc"):
        jobs.append((f"data/{sym}/{sym}_{d}.npz", f"{sym}_{d}.npz"))

out = Path("data/npz")
out.mkdir(parents=True, exist_ok=True)
for remote, local in jobs:
    dest = out / local
    if dest.exists():
        print(f"have  {local}", flush=True)
        continue
    free = os.statvfs("/home/user").f_bavail * os.statvfs("/home/user").f_frsize / 1e9
    if free < 4.0:
        print(f"STOP: only {free:.1f} GB free, refusing to fetch {local}", flush=True)
        sys.exit(1)
    p = hf_hub_download("cardinalai/backtest_data", remote, repo_type="dataset",
                        local_dir="scratchpad/hfdl")
    os.replace(p, dest)
    print(f"got   {local}  {dest.stat().st_size/1e6:.0f} MB  (free {free:.1f} GB)", flush=True)
print("done")
