import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ptbxl.data.labels import LABEL_ORDER
from src.ptbxl.metrics.calibration_metrics import calibration_metrics, sigmoid


def ensure(out_dir):
    out = Path(out_dir)
    for sub in ["calibration", "metrics", "reports", "predictions"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def temp_array(temp):
    if temp["selected_temperature_mode"] == "vector":
        return np.array([temp["temperature"][l] for l in LABEL_ORDER], dtype=float)
    return float(temp["temperature"])


def load_thresholds(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["thresholds"]
    return np.array([d[l] for l in LABEL_ORDER], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--risk-scores", required=True)
    ap.add_argument("--selected-risk", default="Risk_rank_fusion")
    ap.add_argument("--test-predictions", default="outputs_ptbxl/predictions/ptbxl_test_predictions_gpu_seed1.csv")
    ap.add_argument("--out-dir", default="outputs_ptbxl/p5")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    df = pd.read_csv(args.test_predictions)
    y = df[[f"true_{l}" for l in LABEL_ORDER]].values.astype(int)
    logits = df[[f"logit_{l}" for l in LABEL_ORDER]].values.astype(float)
    thresholds = load_thresholds(args.thresholds)
    temp = json.loads(Path(args.temperature).read_text(encoding="utf-8"))
    raw_prob = sigmoid(logits)
    cal_prob = sigmoid(logits / temp_array(temp))
    raw_m = calibration_metrics(y, raw_prob, thresholds)
    cal_m = calibration_metrics(y, cal_prob, thresholds)
    out_df = df[["ecg_id", "patient_id", "strat_fold"] + [f"true_{l}" for l in LABEL_ORDER] + [f"logit_{l}" for l in LABEL_ORDER]].copy()
    for i, l in enumerate(LABEL_ORDER):
        out_df[f"raw_prob_{l}"] = raw_prob[:, i]
        out_df[f"cal_prob_{l}"] = cal_prob[:, i]
        out_df[f"raw_pred_{l}"] = (raw_prob[:, i] >= thresholds[i]).astype(int)
        out_df[f"cal_pred_{l}"] = (cal_prob[:, i] >= thresholds[i]).astype(int)
    risk = pd.read_csv(args.risk_scores)
    if "ecg_id" in risk.columns and "ecg_id" in out_df.columns:
        left = out_df["ecg_id"].astype(str).str.extract(r"(-?\d+)", expand=False)
        right = risk["ecg_id"].astype(str).str.extract(r"(-?\d+)", expand=False)
        if not left.equals(right):
            raise ValueError("Prediction and risk-score rows are not aligned by ecg_id")
    risk_col = args.selected_risk if args.selected_risk in risk.columns else "Risk_rank_fusion"
    out_df[args.selected_risk] = risk[risk_col].values
    out_df.to_csv(out / f"predictions/ptbxl_test_calibrated_predictions_seed{args.seed}.csv", index=False)
    payload = {"raw": raw_m, "calibrated": cal_m, "temperature": temp, "threshold_source": "validation", "test_used_for_threshold_or_temperature": False}
    (out / f"metrics/ptbxl_calibration_all_test_seed{args.seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# P5 Step2 Calibration Report", "", f"Raw NLL: {raw_m['NLL']}", f"Calibrated NLL: {cal_m['NLL']}", f"Raw Brier: {raw_m['Brier']}", f"Calibrated Brier: {cal_m['Brier']}", f"Raw Micro_ECE: {raw_m['Micro_ECE']}", f"Calibrated Micro_ECE: {cal_m['Micro_ECE']}", f"Raw Classwise_ECE_mean: {raw_m['Classwise_ECE_mean']}", f"Calibrated Classwise_ECE_mean: {cal_m['Classwise_ECE_mean']}"]
    (out / "reports/p5_step2_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "raw_nll": raw_m["NLL"], "cal_nll": cal_m["NLL"]}, indent=2))


if __name__ == "__main__":
    main()
