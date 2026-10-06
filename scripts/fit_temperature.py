import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ptbxl.data.labels import LABEL_ORDER
from src.ptbxl.metrics.calibration_metrics import sigmoid


def ensure(out_dir):
    out = Path(out_dir)
    for sub in ["calibration", "metrics", "reports", "predictions"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def load_cal_predictions(path):
    path = Path(path)
    df = pd.read_csv(path)
    y = df[[f"true_{l}" for l in LABEL_ORDER]].values.astype(np.float32)
    logits = df[[f"logit_{l}" for l in LABEL_ORDER]].values.astype(np.float32)
    return y, logits


def bce_np(logits, y):
    p = sigmoid(logits)
    return float(-np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)))


def fit_temp(logits, y, vector=False):
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    init = torch.zeros(lt.shape[1] if vector else 1, requires_grad=True)
    opt = torch.optim.LBFGS([init], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        temp = torch.clamp(torch.exp(init), 0.05, 10.0)
        loss = loss_fn(lt / temp, yt)
        loss.backward()
        return loss

    opt.step(closure)
    temp = torch.clamp(torch.exp(init), 0.05, 10.0).detach().cpu().numpy()
    return temp.tolist() if vector else float(temp[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-predictions", default="outputs_ptbxl/predictions/ptbxl_cal_predictions_gpu_seed1.csv")
    ap.add_argument("--out-dir", default="outputs_ptbxl/p5")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    y, logits = load_cal_predictions(args.cal_predictions)
    before = bce_np(logits, y)
    scalar = fit_temp(logits, y, vector=False)
    vector = fit_temp(logits, y, vector=True)
    nll_scalar = bce_np(logits / scalar, y)
    nll_vector = bce_np(logits / np.asarray(vector)[None, :], y)
    selected = "vector" if nll_vector <= nll_scalar * 0.99 else "scalar"
    scalar_payload = {"seed": args.seed, "mode": "scalar", "temperature": scalar, "source": "calibration", "nll_before": before, "nll_after": nll_scalar}
    vector_payload = {"seed": args.seed, "mode": "vector", "temperature": {l: float(vector[i]) for i, l in enumerate(LABEL_ORDER)}, "source": "calibration", "nll_before": before, "nll_after": nll_vector}
    selected_payload = vector_payload if selected == "vector" else scalar_payload
    selected_payload = dict(selected_payload)
    selected_payload["selected_temperature_mode"] = selected
    selected_payload["test_used_for_selection"] = False
    name_prefix = f"{args.prefix}_" if args.prefix else "ptbxl_"
    (out / f"calibration/{name_prefix}temperature_scalar_seed{args.seed}.json").write_text(json.dumps(scalar_payload, indent=2), encoding="utf-8")
    (out / f"calibration/{name_prefix}temperature_vector_seed{args.seed}.json").write_text(json.dumps(vector_payload, indent=2), encoding="utf-8")
    (out / f"calibration/{name_prefix}temperature_selected_seed{args.seed}.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    report = ["# P5 Step1 Temperature Report", "", f"cal NLL before: {before}", f"scalar T: {scalar}", f"scalar NLL: {nll_scalar}", f"vector T: {vector_payload['temperature']}", f"vector NLL: {nll_vector}", f"selected: {selected}", "test used: NO"]
    (out / "reports/p5_step1_temperature_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "selected": selected, "nll_before": before, "nll_scalar": nll_scalar, "nll_vector": nll_vector}, indent=2))


if __name__ == "__main__":
    main()
