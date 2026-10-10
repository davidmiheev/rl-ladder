"""Push and run a Kaggle script notebook with the official CLI, which honours data sources.

    python scripts/kaggle/kpush.py <slug> <notebook.py> [gpu] [source=<user>/<notebook> ...] [dataset=<user>/<dataset> ...]

Use this instead of kpost.py when a notebook needs inputs: Kaggle's MCP save_notebook accepts
kernelDataSources / datasetDataSources but does not attach them (checked 2026-10-10), while
`kaggle kernels push` mounts them under /kaggle/input/<name>/. The token comes from the
environment (KAGGLE_API_TOKEN), as for the kaggle CLI itself.
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

slug, path, rest = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
gpu = "gpu" in rest
meta = {"id": f"davidmiheev/{slug}", "title": slug, "code_file": path.name, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "kernel_sources": [a.split("=", 1)[1] for a in rest if a.startswith("source=")],
        "dataset_sources": [a.split("=", 1)[1] for a in rest if a.startswith("dataset=")],
        "competition_sources": []}
if gpu:
    meta["machine_shape"] = "NvidiaTeslaT4"
with tempfile.TemporaryDirectory() as d:
    shutil.copy(path, Path(d) / path.name)
    (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta))
    out = subprocess.run([shutil.which("kaggle") or "kaggle", "kernels", "push", "-p", d], capture_output=True, text=True)
print((out.stdout + out.stderr).strip()[-400:])
