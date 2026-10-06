import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_p8, load_json, summarize_values, write_json


def flatten(seed, payload):
    rows = []
    for group, d in payload.items():
        if group == "seed":
            continue
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"seed": seed, "group": group, "metric": f"{k}_{kk}", "value": vv})
            else:
                rows.append({"seed": seed, "group": group, "metric": k, "value": v})
    return rows


def make_summary(df):
    out = {}
    for (group, metric), g in df.groupby(["group", "metric"]):
        vals = pd.to_numeric(g["value"], errors="coerce").dropna().values
        if len(vals):
            out.setdefault(group, {})[metric] = summarize_values(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs_ptbxl/p8")
    args = ap.parse_args()
    out = ensure_p8(args.out_dir)
    rows = []
    for path in sorted((out / "metrics").glob("single_seed_metrics_seed*.json")):
        seed = int(path.stem.split("seed")[-1])
        rows.extend(flatten(seed, load_json(path)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No successful seed metrics found")
    for group, name in [("baseline", "baseline"), ("selective", "selective"), ("calibration", "calibration")]:
        df[df.group == group].to_csv(out / f"tables/table_multiseed_{name}.csv", index=False)
    summary = make_summary(df)
    write_json(out / "metrics/multiseed_summary.json", summary)
    (out / "reports/p8_multiseed_summary_report.md").write_text("# P8 Multiseed Summary Report\n\n" + json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "seeds": sorted(df.seed.unique().tolist())}, indent=2))


if __name__ == "__main__":
    main()
