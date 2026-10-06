import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiment_utils import ensure
from src.ptbxl.metrics.selective_metrics import selective_table, summarize_selective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-scores", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl")
    ap.add_argument("--prefix", default="gpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    df = pd.read_csv(args.risk_scores)
    risk_cols = [
        "MSP_risk",
        "Entropy_risk",
        "Energy_risk",
        "Prototype_risk",
        "RCUF_3",
        "RCUF_4_prototype_ablation",
    ]
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    rows = []
    for risk in risk_cols:
        rows.extend(selective_table(df, risk, coverages))
    table = pd.DataFrame(rows)
    table.to_csv(out / f"metrics/ptbxl_selective_metrics_{args.prefix}_seed{args.seed}.csv", index=False)
    summary = summarize_selective(table)
    checks = {
        "accepted_macro_f1_90_gt_no_reject": any((v["MacroF1@90"] or 0) > (v["NoReject_MacroF1"] or 0) for v in summary.values()),
        "risk_90_lt_risk_100": any((v["Risk@90"] or 1e9) < (v["NoReject_Risk"] or -1) for v in summary.values()),
        "rejected_error_ratio_90_gt_1": any((v["Rejected/Accepted Error Ratio@90"] or 0) > 1.0 for v in summary.values()),
    }
    summary_payload = {"summary": summary, "pass_checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    (out / f"metrics/ptbxl_selective_summary_{args.prefix}_seed{args.seed}.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report = ["# PTB-XL P4 GPU Selective Diagnosis Report", "", f"PASS/FAIL: {summary_payload['status']}", "Score metric and selective decision metric are reported separately.", "", json.dumps(summary_payload, indent=2)]
    (out / f"reports/phase_p4_{args.prefix}_selective_diagnosis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": summary_payload["status"], "pass_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
