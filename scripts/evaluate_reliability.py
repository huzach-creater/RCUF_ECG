import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ptbxl.data.labels import LABEL_ORDER
from src.ptbxl.metrics.calibration_metrics import calibration_metrics


def ensure(out_dir):
    out = Path(out_dir)
    for sub in ["calibration", "metrics", "reports", "predictions"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def metrics_for(df):
    y = df[[f"true_{l}" for l in LABEL_ORDER]].values.astype(int)
    p = df[[f"cal_prob_{l}" for l in LABEL_ORDER]].values.astype(float)
    pred = df[[f"cal_pred_{l}" for l in LABEL_ORDER]].values.astype(int)
    cal = calibration_metrics(y, p, predictions=pred)
    sample_error = 1 - np.array([1.0 if (yt == yp).all() and yt.sum() == 0 else (2 * ((yt & yp).sum()) / max(1, 2 * ((yt & yp).sum()) + ((yp == 1) & (yt == 0)).sum() + ((yp == 0) & (yt == 1)).sum())) for yt, yp in zip(y.astype(bool), pred.astype(bool))])
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "sample_error": float(sample_error.mean()),
        "exact_match_error": float((y != pred).any(axis=1).mean()),
        "NLL": cal["NLL"],
        "Brier": cal["Brier"],
        "Micro_ECE": cal["Micro_ECE"],
        "Classwise_ECE_mean": cal["Classwise_ECE_mean"],
        **{f"ECE_{k}": v for k, v in cal["Classwise_ECE"].items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrated-predictions", required=True)
    ap.add_argument("--risk-scores", required=True)
    ap.add_argument("--selected-risk", default="Risk_rank_fusion")
    ap.add_argument("--out-dir", default="outputs_ptbxl/p5")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    df = pd.read_csv(args.calibrated_predictions)
    risk = pd.read_csv(args.risk_scores)
    left = df["ecg_id"].astype(str).str.extract(r"(-?\d+)", expand=False)
    right = risk["ecg_id"].astype(str).str.extract(r"(-?\d+)", expand=False)
    if not left.equals(right):
        raise ValueError("Calibrated predictions and risk-score rows are not aligned by ecg_id")
    df[args.selected_risk] = risk[args.selected_risk].values
    rows = []
    all_m = metrics_for(df); rows.append({"coverage": 1.0, "subset": "all", **all_m})
    ordered = df.sort_values(args.selected_risk, ascending=True)
    for cov in [0.95, 0.90, 0.80]:
        k = int(round(len(df) * cov))
        acc, rej = ordered.iloc[:k], ordered.iloc[k:]
        rows.append({"coverage": cov, "subset": "accepted", **metrics_for(acc)})
        rows.append({"coverage": cov, "subset": "rejected", **metrics_for(rej)})
    res = pd.DataFrame(rows)
    res.to_csv(out / f"metrics/ptbxl_accepted_rejected_reliability_seed{args.seed}.csv", index=False)
    r90a = res[(res.coverage == 0.9) & (res.subset == "accepted")].iloc[0]
    r90r = res[(res.coverage == 0.9) & (res.subset == "rejected")].iloc[0]
    report = ["# P5 Step3 Accepted Rejected Reliability Report", "", f"Accepted@90 sample error: {r90a.sample_error}", f"Rejected@90 sample error: {r90r.sample_error}", f"Rejected/Accepted ratio: {r90r.sample_error / r90a.sample_error if r90a.sample_error > 0 else None}", f"Accepted@90 operational-decision Micro-ECE: {r90a.Micro_ECE}", f"Rejected@90 operational-decision Micro-ECE: {r90r.Micro_ECE}"]
    (out / "reports/p5_step3_accepted_rejected_reliability_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
