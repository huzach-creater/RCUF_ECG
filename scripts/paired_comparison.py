import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_p8, load_json


METHOD_MAP = {
    "MSP": ("msp_macro_f1_90", "msp_risk90", "msp_aurc"),
    "Entropy": ("entropy_macro_f1_90", "entropy_risk90", "entropy_aurc"),
    "Energy": ("energy_macro_f1_90", "energy_risk90", "energy_aurc"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs_ptbxl/p8")
    args = ap.parse_args()
    out = ensure_p8(args.out_dir)
    seeds = []
    vals = {}
    for path in sorted((out / "metrics").glob("single_seed_metrics_seed*.json")):
        payload = load_json(path)
        seed = payload["seed"]
        seeds.append(seed)
        vals[seed] = payload["selective"]
    rows = []
    for method, keys in METHOD_MAP.items():
        for metric, rcuf_key, other_key in [
            ("Macro-F1@90", "rcuf_macro_f1_90", keys[0]),
            ("Risk@90", "rcuf_risk90", keys[1]),
            ("AURC", "rcuf_aurc", keys[2]),
        ]:
            diffs = np.array([vals[s][rcuf_key] - vals[s][other_key] for s in seeds], dtype=float)
            rows.append({
                "comparison": f"RCUF vs {method}",
                "metric": metric,
                "mean_difference": float(diffs.mean()),
                "std_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "seed_differences": ";".join(f"{s}:{d}" for s, d in zip(seeds, diffs)),
                "note": "Statistical tests are exploratory due to limited number of seeds.",
            })
    pd.DataFrame(rows).to_csv(out / "tables/table_paired_comparison.csv", index=False)
    (out / "reports/p8_paired_comparison_report.md").write_text("# P8 Paired Comparison Report\n\nStatistical tests are exploratory due to limited number of seeds.", encoding="utf-8")
    print(json.dumps({"status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
