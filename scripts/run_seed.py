import argparse
import json
import sys
from pathlib import Path

import yaml

from pipeline_utils import copy_if_exists, ensure_p8, load_json, run, write_json


def extract_seed_metrics(out, seed):
    baseline = load_json(out / f"metrics/baseline_metrics_seed{seed}.json")
    b = baseline["test_metrics_threshold_opt"]
    selective = load_json(out / f"metrics/ptbxl_selective_summary_seed{seed}_seed{seed}.json")["summary"]
    r = selective["RCUF_3"]
    cal = load_json(out / f"metrics/ptbxl_calibration_all_test_seed{seed}.json")
    metrics = {
        "seed": seed,
        "baseline": {
            "macro_auc": b["macro_auc"],
            "micro_auc": b["micro_auc"],
            "macro_auprc": b["macro_auprc"],
            "macro_f1": b["macro_f1"],
            "micro_f1": b["micro_f1"],
            "per_class_auc": b["per_class_auc"],
            "per_class_f1": b["per_class_f1"],
        },
        "selective": {
            "no_reject_macro_f1": r["NoReject_MacroF1"],
            "no_reject_micro_f1": r["NoReject_MicroF1"],
            "no_reject_risk": r["NoReject_Risk"],
            "rcuf_macro_f1_90": r["MacroF1@90"],
            "rcuf_risk90": r["Risk@90"],
            "rcuf_aurc": r["AURC"],
            "rcuf_ratio90": r["Rejected/Accepted Error Ratio@90"],
            "msp_macro_f1_90": selective["MSP_risk"]["MacroF1@90"],
            "msp_risk90": selective["MSP_risk"]["Risk@90"],
            "msp_aurc": selective["MSP_risk"]["AURC"],
            "entropy_macro_f1_90": selective["Entropy_risk"]["MacroF1@90"],
            "entropy_risk90": selective["Entropy_risk"]["Risk@90"],
            "entropy_aurc": selective["Entropy_risk"]["AURC"],
            "energy_macro_f1_90": selective["Energy_risk"]["MacroF1@90"],
            "energy_risk90": selective["Energy_risk"]["Risk@90"],
            "energy_aurc": selective["Energy_risk"]["AURC"],
        },
        "calibration": {
            "raw_nll": cal["raw"]["NLL"],
            "calibrated_nll": cal["calibrated"]["NLL"],
            "raw_brier": cal["raw"]["Brier"],
            "calibrated_brier": cal["calibrated"]["Brier"],
            "raw_micro_ece": cal["raw"]["Micro_ECE"],
            "calibrated_micro_ece": cal["calibrated"]["Micro_ECE"],
            "raw_classwise_ece_mean": cal["raw"]["Classwise_ECE_mean"],
            "calibrated_classwise_ece_mean": cal["calibrated"]["Classwise_ECE_mean"],
        },
    }
    write_json(out / f"metrics/single_seed_metrics_seed{seed}.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl/p8")
    args = ap.parse_args()
    out = ensure_p8(args.out_dir)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed = args.seed
    prefix = f"seed{seed}"
    data, train_cfg, model_cfg = cfg["data"], cfg["train"]["config"], cfg["model"]["config"]
    py = sys.executable
    status = {"seed": seed, "status": "PASS", "steps": []}

    try:
        run([
            py, "scripts/train_backbone.py",
            "--train-data", data["train"], "--val-data", data["val"],
            "--config", train_cfg, "--model-config", model_cfg, "--out-dir", str(out),
            "--checkpoint-name", f"ptbxl_resnet1d_seed{seed}.pt", "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/train_seed{seed}.log")
        status["steps"].append("train")

        ckpt = out / f"checkpoints/ptbxl_resnet1d_seed{seed}.pt"
        run([
            py, "scripts/evaluate_backbone.py",
            "--checkpoint", str(ckpt), "--val-data", data["val"], "--cal-data", data["cal"], "--test-data", data["test"],
            "--out-dir", str(out), "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/eval_seed{seed}.log")
        copy_if_exists(out / f"metrics/ptbxl_resnet_{prefix}_baseline_metrics_seed{seed}.json", out / f"metrics/baseline_metrics_seed{seed}.json")
        copy_if_exists(out / f"calibration/ptbxl_label_thresholds_{prefix}_seed{seed}.json", out / f"calibration/label_thresholds_seed{seed}.json")
        copy_if_exists(out / f"predictions/ptbxl_test_predictions_{prefix}_seed{seed}.csv", out / f"predictions/ptbxl_test_predictions_seed{seed}.csv")
        copy_if_exists(out / f"predictions/ptbxl_cal_predictions_{prefix}_seed{seed}.csv", out / f"predictions/ptbxl_cal_predictions_seed{seed}.csv")
        status["steps"].append("baseline_eval")

        run([
            py, "scripts/fit_risk_scores.py",
            "--checkpoint", str(ckpt), "--train-data", data["train"], "--cal-data", data["cal"], "--test-data", data["test"],
            "--thresholds", str(out / f"calibration/label_thresholds_seed{seed}.json"),
            "--out-dir", str(out), "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/risk_seed{seed}.log")
        copy_if_exists(out / f"predictions/ptbxl_test_risk_scores_{prefix}_seed{seed}.csv", out / f"predictions/ptbxl_test_risk_scores_seed{seed}.csv")
        copy_if_exists(out / f"predictions/ptbxl_cal_risk_scores_{prefix}_seed{seed}.csv", out / f"predictions/ptbxl_cal_risk_scores_seed{seed}.csv")
        copy_if_exists(out / f"calibration/ptbxl_risk_reference_{prefix}_seed{seed}.npz", out / f"calibration/risk_reference_seed{seed}.npz")
        copy_if_exists(out / f"calibration/ptbxl_prototypes_{prefix}_seed{seed}.npz", out / f"calibration/prototypes_seed{seed}.npz")
        status["steps"].append("risk")

        run([
            py, "scripts/evaluate_selective.py",
            "--risk-scores", str(out / f"predictions/ptbxl_test_risk_scores_seed{seed}.csv"),
            "--out-dir", str(out), "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/selective_seed{seed}.log")
        # Keep a compact alias for aggregation.
        copy_if_exists(out / f"metrics/ptbxl_selective_summary_{prefix}_seed{seed}.json", out / f"metrics/selective_metrics_seed{seed}.json")
        status["steps"].append("selective")

        run([
            py, "scripts/fit_temperature.py",
            "--cal-predictions", str(out / f"predictions/ptbxl_cal_predictions_seed{seed}.csv"),
            "--out-dir", str(out), "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/temp_seed{seed}.log")
        copy_if_exists(out / f"calibration/{prefix}_temperature_selected_seed{seed}.json", out / f"calibration/temperature_selected_seed{seed}.json")
        run([
            py, "scripts/evaluate_calibration.py",
            "--temperature", str(out / f"calibration/temperature_selected_seed{seed}.json"),
            "--thresholds", str(out / f"calibration/label_thresholds_seed{seed}.json"),
            "--risk-scores", str(out / f"predictions/ptbxl_test_risk_scores_seed{seed}.csv"),
            "--selected-risk", "RCUF_3", "--test-predictions", str(out / f"predictions/ptbxl_test_predictions_seed{seed}.csv"),
            "--out-dir", str(out), "--prefix", prefix, "--seed", str(seed),
        ], out / f"logs/calibration_seed{seed}.log")
        copy_if_exists(out / f"metrics/ptbxl_calibration_all_test_seed{seed}.json", out / f"metrics/calibration_metrics_seed{seed}.json")
        status["steps"].append("calibration")

        run([
            py, "scripts/evaluate_reliability.py",
            "--calibrated-predictions", str(out / f"predictions/ptbxl_test_calibrated_predictions_seed{seed}.csv"),
            "--risk-scores", str(out / f"predictions/ptbxl_test_risk_scores_seed{seed}.csv"),
            "--selected-risk", "RCUF_3", "--out-dir", str(out), "--seed", str(seed),
        ], out / f"logs/reliability_seed{seed}.log")
        status["steps"].append("reliability")

        metrics = extract_seed_metrics(out, seed)
        report = ["# P8 Single Seed Report", "", f"seed: {seed}", f"status: PASS", json.dumps(metrics, indent=2)]
        (out / f"reports/single_seed_report_seed{seed}.md").write_text("\n".join(report), encoding="utf-8")
    except Exception as exc:
        status["status"] = "FAIL"
        status["error"] = repr(exc)
        (out / f"reports/single_seed_report_seed{seed}.md").write_text("# P8 Single Seed Report\n\n" + json.dumps(status, indent=2), encoding="utf-8")
        raise
    finally:
        write_json(out / f"metrics/single_seed_status_seed{seed}.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
