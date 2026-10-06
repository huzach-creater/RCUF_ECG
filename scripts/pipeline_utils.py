import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def ensure_p8(out_dir):
    out = Path(out_dir)
    for sub in ["checkpoints", "predictions", "metrics", "tables", "reports", "logs", "calibration", "embeddings"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd, log_path=None):
    print("RUN:", " ".join(map(str, cmd)))
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(cmd)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def copy_if_exists(src, dst):
    src, dst = Path(src), Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def summarize_values(vals, bootstrap=1000, seed=1):
    arr = np.asarray(vals, dtype=float)
    out = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(bootstrap)]
    out["ci95_lower"] = float(np.percentile(boots, 2.5))
    out["ci95_upper"] = float(np.percentile(boots, 97.5))
    return out


def metric_row(seed, metric, value):
    return {"seed": seed, "metric": metric, "value": value}
