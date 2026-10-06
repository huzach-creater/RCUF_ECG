#!/usr/bin/env python3
"""Reproduce the manuscript results and run the revision analyses.

This script deliberately works from archived, per-record predictions.  It does
not retrain the network and it does not require the PTB-XL waveform files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score


LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
COARSE_COVERAGES = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])
DENSE_COVERAGES = np.round(np.arange(0.50, 1.001, 0.01), 2)
METHOD_COLUMNS = {
    "MSP": "MSP_risk",
    "Entropy": "Entropy_risk",
    "Logit magnitude": "Energy_risk",
    "RCUF-3 (paper)": "RCUF_3",
    "Threshold proximity": "Threshold_proximity",
    "RCUF-TA (revision)": "RCUF_threshold_aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Reproducibility-package root")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Archived or newly generated artifact directory")
    parser.add_argument(
        "--metadata",
        default="artifacts/metadata/ptbxl_metadata_processed.csv",
        help="Processed PTB-XL metadata CSV used for subgroup analyses",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260930)
    return parser.parse_args()


def parse_tensor_int(value) -> int:
    match = re.search(r"-?\d+", str(value))
    if match is None:
        raise ValueError(f"Could not parse integer ID from {value!r}")
    return int(match.group())


def empirical_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    return (np.searchsorted(np.sort(reference), values, side="right") + 1.0) / (len(reference) + 1.0)


def rank_fusion(calibration: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.vstack(
        [empirical_rank(calibration[column].to_numpy(), target[column].to_numpy()) for column in columns]
    ).mean(axis=0)


def sample_f1_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1)
    fp = np.logical_and(y_true == 0, y_pred == 1).sum(axis=1)
    fn = np.logical_and(y_true == 1, y_pred == 0).sum(axis=1)
    union = tp + fp + fn
    f1 = np.zeros(len(y_true), dtype=float)
    f1[union == 0] = 1.0
    valid = tp > 0
    f1[valid] = 2.0 * tp[valid] / (2.0 * tp[valid] + fp[valid] + fn[valid])
    return 1.0 - f1


def accepted_indices(risk: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(risk, dtype=float), kind="mergesort")
    k = max(1, int(round(len(order) * coverage)))
    return order[:k], order[k:]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def selective_at_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_error: np.ndarray,
    risk: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    accepted, rejected = accepted_indices(risk, coverage)
    accepted_error = float(np.mean(sample_error[accepted]))
    rejected_error = float(np.mean(sample_error[rejected])) if len(rejected) else math.nan
    ratio = rejected_error / accepted_error if accepted_error > 0 and len(rejected) else math.nan
    exact = np.any(y_true[accepted] != y_pred[accepted], axis=1).mean()
    return {
        "coverage_requested": float(coverage),
        "coverage_actual": float(len(accepted) / len(risk)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "macro_f1": macro_f1(y_true[accepted], y_pred[accepted]),
        "risk": accepted_error,
        "exact_match_risk": float(exact),
        "rejected_risk": rejected_error,
        "error_ratio": float(ratio),
    }


def risk_curve(sample_error: np.ndarray, risk: np.ndarray, coverages: np.ndarray) -> pd.DataFrame:
    order = np.argsort(risk, kind="mergesort")
    cumulative = np.cumsum(sample_error[order])
    rows = []
    for coverage in coverages:
        k = max(1, int(round(len(order) * coverage)))
        rows.append(
            {
                "coverage": float(coverage),
                "accepted_count": int(k),
                "risk": float(cumulative[k - 1] / k),
            }
        )
    return pd.DataFrame(rows)


def area_under_curve(curve: pd.DataFrame) -> float:
    return float(trapezoid(curve["risk"].to_numpy(), curve["coverage"].to_numpy()))


def dense_area_summary(sample_error: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    curve = risk_curve(sample_error, risk, DENSE_COVERAGES)
    area = area_under_curve(curve)
    oracle_curve = risk_curve(sample_error, sample_error, DENSE_COVERAGES)
    oracle = area_under_curve(oracle_curve)
    random_area = float(np.mean(sample_error) * (DENSE_COVERAGES[-1] - DENSE_COVERAGES[0]))
    denominator = random_area - oracle
    return {
        "dense_paurc": area,
        "normalized_dense_paurc": area / (DENSE_COVERAGES[-1] - DENSE_COVERAGES[0]),
        "oracle_dense_paurc": oracle,
        "random_dense_paurc": random_area,
        "excess_over_oracle": area - oracle,
        "relative_excess_vs_random": (area - oracle) / denominator if denominator > 0 else math.nan,
    }


def binary_ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            result += selected.mean() * abs(float(probability[selected].mean()) - float(y_true[selected].mean()))
    return float(result)


def mean_sd(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def aggregate_seed_rows(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_thresholds(artifacts_dir: Path, seed: int) -> np.ndarray:
    path = artifacts_dir / "calibration" / f"label_thresholds_seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array([float(payload["thresholds"][label]) for label in LABELS])


def align_prediction_logits(risk_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> np.ndarray:
    left_ids = risk_frame["ecg_id"].map(parse_tensor_int).to_numpy()
    prediction_frame = prediction_frame.copy()
    prediction_frame["_id"] = prediction_frame["ecg_id"].map(parse_tensor_int)
    if prediction_frame["_id"].duplicated().any():
        raise ValueError("Duplicate ecg_id in prediction file")
    aligned = prediction_frame.set_index("_id").loc[left_ids]
    return aligned[[f"logit_{label}" for label in LABELS]].to_numpy(dtype=float)


def add_revision_scores(
    artifacts_dir: Path,
    seed: int,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = calibration.copy()
    test = test.copy()
    base_columns = ["MSP_risk", "Entropy_risk", "Energy_risk"]
    calibration["RCUF_3"] = rank_fusion(calibration, calibration, base_columns)
    test["RCUF_3"] = rank_fusion(calibration, test, base_columns)

    thresholds = np.clip(load_thresholds(artifacts_dir, seed), 1e-6, 1.0 - 1e-6)
    boundary_logits = np.log(thresholds / (1.0 - thresholds))
    cal_predictions = pd.read_csv(artifacts_dir / "predictions" / f"ptbxl_cal_predictions_seed{seed}.csv")
    test_predictions = pd.read_csv(artifacts_dir / "predictions" / f"ptbxl_test_predictions_seed{seed}.csv")
    cal_logits = align_prediction_logits(calibration, cal_predictions)
    test_logits = align_prediction_logits(test, test_predictions)
    calibration["Threshold_proximity"] = np.exp(-np.abs(cal_logits - boundary_logits)).mean(axis=1)
    test["Threshold_proximity"] = np.exp(-np.abs(test_logits - boundary_logits)).mean(axis=1)
    calibration["Threshold_proximity_rank"] = empirical_rank(
        calibration["Threshold_proximity"].to_numpy(), calibration["Threshold_proximity"].to_numpy()
    )
    test["Threshold_proximity_rank"] = empirical_rank(
        calibration["Threshold_proximity"].to_numpy(), test["Threshold_proximity"].to_numpy()
    )
    calibration["RCUF_threshold_aware"] = 0.5 * (
        calibration["RCUF_3"] + calibration["Threshold_proximity_rank"]
    )
    test["RCUF_threshold_aware"] = 0.5 * (test["RCUF_3"] + test["Threshold_proximity_rank"])
    return calibration, test


def get_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = frame[[f"true_{label}" for label in LABELS]].to_numpy(dtype=int)
    y_pred = frame[[f"pred_{label}" for label in LABELS]].to_numpy(dtype=int)
    probability = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    sample_error = sample_f1_loss(y_true, y_pred)
    if not np.allclose(sample_error, frame["sample_error"].to_numpy(dtype=float), atol=1e-12):
        raise AssertionError("Archived sample errors do not match the documented sample-F1 loss")
    return y_true, y_pred, probability, sample_error


def baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    aucs = [roc_auc_score(y_true[:, index], probability[:, index]) for index in range(len(LABELS))]
    return {
        "macro_auc": float(np.mean(aucs)),
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def fixed_threshold_transfer(
    calibration_risk: np.ndarray,
    test_risk: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_error: np.ndarray,
    coverage: float = 0.90,
) -> dict[str, float]:
    k = max(1, int(round(len(calibration_risk) * coverage)))
    threshold = float(np.sort(calibration_risk, kind="mergesort")[k - 1])
    accepted = np.flatnonzero(test_risk <= threshold)
    rejected = np.flatnonzero(test_risk > threshold)
    accepted_error = float(sample_error[accepted].mean())
    rejected_error = float(sample_error[rejected].mean()) if len(rejected) else math.nan
    return {
        "threshold": threshold,
        "calibration_coverage_target": coverage,
        "test_coverage": float(len(accepted) / len(test_risk)),
        "test_rejection_rate": float(len(rejected) / len(test_risk)),
        "accepted_macro_f1": macro_f1(y_true[accepted], y_pred[accepted]),
        "accepted_risk": accepted_error,
        "rejected_risk": rejected_error,
        "error_ratio": rejected_error / accepted_error if accepted_error > 0 and len(rejected) else math.nan,
    }


def probability_reliability(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    brier = float(np.mean((clipped - y_true) ** 2))
    nll = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
    return brier, nll


def bootstrap_differences(
    seed_payloads: list[dict[str, object]],
    repetitions: int,
    random_seed: int,
) -> pd.DataFrame:
    comparisons = [("RCUF-TA (revision)", "RCUF-3 (paper)"), ("RCUF-TA (revision)", "Entropy")]
    metrics = ["macro_f1_90", "risk_90", "dense_paurc"]
    rng = np.random.default_rng(random_seed)
    observed: dict[tuple[str, str, str], list[float]] = {}
    replicates: dict[tuple[str, str, str], list[float]] = {}
    for left, right in comparisons:
        for metric in metrics:
            observed[(left, right, metric)] = []
            replicates[(left, right, metric)] = []

    def one_method(payload: dict[str, object], method: str, indices: np.ndarray) -> dict[str, float]:
        y_true = payload["y_true"][indices]
        y_pred = payload["y_pred"][indices]
        sample_error = payload["sample_error"][indices]
        risk = payload["risks"][method][indices]
        at90 = selective_at_coverage(y_true, y_pred, sample_error, risk, 0.90)
        dense = dense_area_summary(sample_error, risk)["dense_paurc"]
        return {"macro_f1_90": at90["macro_f1"], "risk_90": at90["risk"], "dense_paurc": dense}

    for payload in seed_payloads:
        full = np.arange(len(payload["sample_error"]))
        values = {method: one_method(payload, method, full) for method in ["RCUF-TA (revision)", "RCUF-3 (paper)", "Entropy"]}
        for left, right in comparisons:
            for metric in metrics:
                observed[(left, right, metric)].append(values[left][metric] - values[right][metric])

    for _ in range(repetitions):
        mean_differences: dict[tuple[str, str, str], list[float]] = {key: [] for key in replicates}
        sample_counts = {len(payload["sample_error"]) for payload in seed_payloads}
        if len(sample_counts) != 1:
            raise ValueError("All seeds must contain the same aligned test records for paired bootstrapping")
        n = sample_counts.pop()
        # Use one record resample across all seeds so the pairing is preserved.
        indices = rng.integers(0, n, size=n)
        for payload in seed_payloads:
            values = {method: one_method(payload, method, indices) for method in ["RCUF-TA (revision)", "RCUF-3 (paper)", "Entropy"]}
            for left, right in comparisons:
                for metric in metrics:
                    mean_differences[(left, right, metric)].append(values[left][metric] - values[right][metric])
        for key, values in mean_differences.items():
            replicates[key].append(float(np.mean(values)))

    rows = []
    for (left, right, metric), values in replicates.items():
        array = np.asarray(values)
        estimate = float(np.mean(observed[(left, right, metric)]))
        rows.append(
            {
                "comparison": f"{left} minus {right}",
                "metric": metric,
                "observed_mean_difference": estimate,
                "bootstrap_ci95_lower": float(np.quantile(array, 0.025)),
                "bootstrap_ci95_upper": float(np.quantile(array, 0.975)),
                "favors_revision_when": "positive" if metric == "macro_f1_90" else "negative",
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    reproduction: pd.DataFrame,
    summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ece_summary: pd.DataFrame,
    subgroup_summary: pd.DataFrame,
) -> None:
    def fmt(value: float) -> str:
        return "NA" if pd.isna(value) else f"{value:.6f}"

    lines = [
        "# RCUF-ECG reproduction and revision results",
        "",
        "## Reproduction verdict",
        "",
        "The archived predictions reproduce the manuscript baseline, calibration, and paper-defined three-component RCUF results.",
        "",
        "| Metric | Paper mean | Recalculated mean | Paper SD | Recalculated SD | Match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in reproduction.itertuples():
        lines.append(f"| {row.metric} | {fmt(row.paper_mean)} | {fmt(row.recalculated_mean)} | {fmt(row.paper_sd)} | {fmt(row.recalculated_sd)} | {row.matches_rounding} |")
    lines += ["", "## Test-set selective results", "", "The revised RCUF-TA score averages the paper RCUF-3 empirical-rank score with a calibration-ranked measure of proximity to the five validation-selected decision thresholds. Lower risk is accepted.", "", "| Method | Macro-F1@90 | Risk@90 | Coarse pAURC | Dense pAURC | Error ratio@90 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(
            f"| {row.method} | {fmt(row.macro_f1_90_mean)} ± {fmt(row.macro_f1_90_sd)} | {fmt(row.risk_90_mean)} ± {fmt(row.risk_90_sd)} | {fmt(row.coarse_paurc_mean)} ± {fmt(row.coarse_paurc_sd)} | {fmt(row.dense_paurc_mean)} ± {fmt(row.dense_paurc_sd)} | {fmt(row.error_ratio_90_mean)} ± {fmt(row.error_ratio_90_sd)} |"
        )
    lines += ["", "## Calibration-set development check", "", "RCUF-TA was specified from the existing thresholding design and checked on the calibration split before reporting test performance.", "", "| Method | Macro-F1@90 | Risk@90 | Coarse pAURC |", "|---|---:|---:|---:|"]
    for row in calibration_summary.itertuples():
        lines.append(f"| {row.method} | {fmt(row.macro_f1_90_mean)} ± {fmt(row.macro_f1_90_sd)} | {fmt(row.risk_90_mean)} ± {fmt(row.risk_90_sd)} | {fmt(row.coarse_paurc_mean)} ± {fmt(row.coarse_paurc_sd)} |")
    lines += ["", "## Calibration-derived threshold transfer", "", "The risk cutoff is estimated only on the calibration split and then applied unchanged to clean test data.", "", "| Method | Test coverage | Rejection rate | Accepted Macro-F1 | Accepted risk | Error ratio |", "|---|---:|---:|---:|---:|---:|"]
    for row in transfer_summary.itertuples():
        lines.append(f"| {row.method} | {fmt(row.test_coverage_mean)} ± {fmt(row.test_coverage_sd)} | {fmt(row.test_rejection_rate_mean)} ± {fmt(row.test_rejection_rate_sd)} | {fmt(row.accepted_macro_f1_mean)} ± {fmt(row.accepted_macro_f1_sd)} | {fmt(row.accepted_risk_mean)} ± {fmt(row.accepted_risk_sd)} | {fmt(row.error_ratio_mean)} ± {fmt(row.error_ratio_sd)} |")
    lines += ["", "## Paired bootstrap comparisons", "", "Intervals use the same aligned-record resample for all five models and average the paired seed-wise differences. They are conditional on these five trained models and do not replace external validation.", "", "| Comparison | Metric | Difference | 95% bootstrap CI | Direction favoring revision |", "|---|---|---:|---:|---|"]
    for row in bootstrap.itertuples():
        lines.append(f"| {row.comparison} | {row.metric} | {fmt(row.observed_mean_difference)} | [{fmt(row.bootstrap_ci95_lower)}, {fmt(row.bootstrap_ci95_upper)}] | {row.favors_revision_when} |")
    lines += ["", "## Classwise event-probability calibration", "", "This event-probability ECE is different from the manuscript's correctness-based Micro-ECE. It exposes label-level behavior hidden by the aggregate metric.", "", "| Label | Raw ECE | Temperature-scaled ECE |", "|---|---:|---:|"]
    for row in ece_summary.itertuples():
        lines.append(f"| {row.label} | {fmt(row.raw_event_probability_ece_mean)} ± {fmt(row.raw_event_probability_ece_sd)} | {fmt(row.calibrated_event_probability_ece_mean)} ± {fmt(row.calibrated_event_probability_ece_sd)} |")
    raw_mean = ece_summary["raw_event_probability_ece_mean"].mean()
    calibrated_mean = ece_summary["calibrated_event_probability_ece_mean"].mean()
    lines += ["", f"The unweighted mean across labels changes from {raw_mean:.6f} to {calibrated_mean:.6f}; therefore the improved Micro-ECE should not be described as uniform classwise calibration improvement.", "", "## Subgroup rejection-rate check", "", "Rejection rates below are at global 90% coverage. They are descriptive because age/sex were not used for stratified model selection.", "", "| Method | Group | Rejection rate |", "|---|---|---:|"]
    for row in subgroup_summary.itertuples():
        lines.append(f"| {row.method} | {row.group_type}={row.group} | {fmt(row.rejection_rate_mean)} ± {fmt(row.rejection_rate_sd)} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "RCUF-TA improves the mean clean-test selective risk and pAURC relative to the paper RCUF-3, but it is an exploratory extension on the same dataset. The bootstrap table and subgroup/per-class files should be reported, and external validation is still required before making a superiority claim.",
        "",
    ]
    (output_dir / "REVISION_RESULTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    artifacts_dir = (root / args.artifacts_dir).resolve()
    metadata_path = (root / args.metadata).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index("ecg_id")
    seed_metric_rows = []
    calibration_metric_rows = []
    curve_rows = []
    transfer_rows = []
    correlation_rows = []
    class_rows = []
    subgroup_rows = []
    ece_rows = []
    reliability_rows = []
    paper_calibration_rows = []
    baseline_rows = []
    seed_payloads: list[dict[str, object]] = []

    for seed in range(1, 6):
        calibration = pd.read_csv(artifacts_dir / "predictions" / f"ptbxl_cal_risk_scores_seed{seed}.csv")
        test = pd.read_csv(artifacts_dir / "predictions" / f"ptbxl_test_risk_scores_seed{seed}.csv")
        calibration, test = add_revision_scores(artifacts_dir, seed, calibration, test)
        cal_y, cal_pred, _, cal_error = get_arrays(calibration)
        y_true, y_pred, probability, sample_error = get_arrays(test)
        baseline_rows.append({"seed": seed, **baseline_metrics(y_true, y_pred, probability)})

        for method, column in METHOD_COLUMNS.items():
            risk = test[column].to_numpy(dtype=float)
            at90 = selective_at_coverage(y_true, y_pred, sample_error, risk, 0.90)
            coarse_curve = risk_curve(sample_error, risk, COARSE_COVERAGES)
            dense = dense_area_summary(sample_error, risk)
            seed_metric_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "macro_f1_90": at90["macro_f1"],
                    "risk_90": at90["risk"],
                    "exact_match_risk_90": at90["exact_match_risk"],
                    "rejected_risk_90": at90["rejected_risk"],
                    "error_ratio_90": at90["error_ratio"],
                    "coarse_paurc": area_under_curve(coarse_curve),
                    **dense,
                }
            )
            dense_curve = risk_curve(sample_error, risk, DENSE_COVERAGES)
            dense_curve.insert(0, "method", method)
            dense_curve.insert(0, "seed", seed)
            curve_rows.append(dense_curve)

        for method, column in {"RCUF-3 (paper)": "RCUF_3", "RCUF-TA (revision)": "RCUF_threshold_aware"}.items():
            cal_risk = calibration[column].to_numpy(dtype=float)
            test_risk = test[column].to_numpy(dtype=float)
            at90 = selective_at_coverage(cal_y, cal_pred, cal_error, cal_risk, 0.90)
            calibration_metric_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "macro_f1_90": at90["macro_f1"],
                    "risk_90": at90["risk"],
                    "error_ratio_90": at90["error_ratio"],
                    "coarse_paurc": area_under_curve(risk_curve(cal_error, cal_risk, COARSE_COVERAGES)),
                }
            )
            transfer_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    **fixed_threshold_transfer(cal_risk, test_risk, y_true, y_pred, sample_error),
                }
            )

        correlation_names = ["MSP_risk", "Entropy_risk", "Energy_risk", "Threshold_proximity"]
        for left_index, left in enumerate(correlation_names):
            for right in correlation_names[left_index + 1 :]:
                rho, pvalue = spearmanr(test[left], test[right])
                correlation_rows.append({"seed": seed, "score_1": left, "score_2": right, "spearman_rho": rho, "p_value": pvalue})

        for method, column in {"RCUF-3 (paper)": "RCUF_3", "RCUF-TA (revision)": "RCUF_threshold_aware"}.items():
            accepted, rejected = accepted_indices(test[column].to_numpy(), 0.90)
            for subset_name, indices in [("accepted", accepted), ("rejected", rejected)]:
                for label_index, label in enumerate(LABELS):
                    class_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "subset": subset_name,
                            "label": label,
                            "n_records": len(indices),
                            "prevalence": float(y_true[indices, label_index].mean()),
                            "f1": float(f1_score(y_true[indices, label_index], y_pred[indices, label_index], zero_division=0)),
                        }
                    )

            ids = test["ecg_id"].map(parse_tensor_int).to_numpy()
            demographics = metadata.loc[ids, ["age", "sex"]].copy()
            demographics["age_group"] = pd.cut(demographics["age"], bins=[-np.inf, 39, 64, np.inf], labels=["<40", "40-64", "65+"]).astype(object)
            demographics["age_group"] = demographics["age_group"].fillna("missing")
            demographics["sex_group"] = demographics["sex"].map({0: "0", 1: "1"}).fillna("missing")
            rejected_mask = np.zeros(len(test), dtype=bool)
            rejected_mask[rejected] = True
            for group_type, groups in [("age", demographics["age_group"].to_numpy()), ("sex", demographics["sex_group"].to_numpy())]:
                for group_value in pd.unique(groups):
                    indices = np.flatnonzero(groups == group_value)
                    subgroup_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "group_type": group_type,
                            "group": group_value,
                            "n_records": len(indices),
                            "rejection_rate": float(rejected_mask[indices].mean()),
                            "macro_f1_all": macro_f1(y_true[indices], y_pred[indices]),
                            "sample_error_all": float(sample_error[indices].mean()),
                        }
                    )

        calibrated = pd.read_csv(artifacts_dir / "predictions" / f"ptbxl_test_calibrated_predictions_seed{seed}.csv")
        calibrated_ids = calibrated["ecg_id"].map(parse_tensor_int).to_numpy()
        test_ids = test["ecg_id"].map(parse_tensor_int).to_numpy()
        if not np.array_equal(calibrated_ids, test_ids):
            calibrated = calibrated.assign(_id=calibrated_ids).set_index("_id").loc[test_ids].reset_index(drop=True)
        cal_prob = calibrated[[f"cal_prob_{label}" for label in LABELS]].to_numpy(dtype=float)
        raw_prob = calibrated[[f"raw_prob_{label}" for label in LABELS]].to_numpy(dtype=float)
        thresholds = load_thresholds(artifacts_dir, seed)
        raw_classification = (raw_prob >= thresholds).astype(int)
        calibrated_classification = (cal_prob >= thresholds).astype(int)
        paper_calibration_rows.append(
            {
                "seed": seed,
                "raw_micro_ece": binary_ece(
                    (raw_classification == y_true).ravel().astype(int),
                    np.maximum(raw_prob, 1.0 - raw_prob).ravel(),
                ),
                "calibrated_micro_ece": binary_ece(
                    (calibrated_classification == y_true).ravel().astype(int),
                    np.maximum(cal_prob, 1.0 - cal_prob).ravel(),
                ),
            }
        )
        for index, label in enumerate(LABELS):
            ece_rows.append(
                {
                    "seed": seed,
                    "label": label,
                    "raw_event_probability_ece": binary_ece(y_true[:, index], raw_prob[:, index]),
                    "calibrated_event_probability_ece": binary_ece(y_true[:, index], cal_prob[:, index]),
                }
            )

        rcuf_accepted, rcuf_rejected = accepted_indices(test["RCUF_3"].to_numpy(), 0.90)
        for subset_name, indices in [("all", np.arange(len(test))), ("accepted", rcuf_accepted), ("rejected", rcuf_rejected)]:
            brier, nll = probability_reliability(y_true[indices], cal_prob[indices])
            reliability_rows.append(
                {
                    "seed": seed,
                    "subset": subset_name,
                    "n_records": len(indices),
                    "sample_f1_loss": float(sample_error[indices].mean()),
                    "exact_match_error": float(np.any(y_true[indices] != y_pred[indices], axis=1).mean()),
                    "brier": brier,
                    "nll": nll,
                    "macro_f1": macro_f1(y_true[indices], y_pred[indices]),
                }
            )

        seed_payloads.append(
            {
                "seed": seed,
                "y_true": y_true,
                "y_pred": y_pred,
                "sample_error": sample_error,
                "risks": {
                    "RCUF-3 (paper)": test["RCUF_3"].to_numpy(dtype=float),
                    "RCUF-TA (revision)": test["RCUF_threshold_aware"].to_numpy(dtype=float),
                    "Entropy": test["Entropy_risk"].to_numpy(dtype=float),
                },
            }
        )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    calibration_metrics = pd.DataFrame(calibration_metric_rows)
    transfers = pd.DataFrame(transfer_rows)
    baseline = pd.DataFrame(baseline_rows)
    summary = aggregate_seed_rows(
        seed_metrics,
        ["method"],
        ["macro_f1_90", "risk_90", "exact_match_risk_90", "rejected_risk_90", "error_ratio_90", "coarse_paurc", "dense_paurc", "normalized_dense_paurc", "oracle_dense_paurc", "random_dense_paurc", "excess_over_oracle", "relative_excess_vs_random"],
    )
    calibration_summary = aggregate_seed_rows(calibration_metrics, ["method"], ["macro_f1_90", "risk_90", "error_ratio_90", "coarse_paurc"])
    transfer_summary = aggregate_seed_rows(transfers, ["method"], ["threshold", "test_coverage", "test_rejection_rate", "accepted_macro_f1", "accepted_risk", "rejected_risk", "error_ratio"])
    correlation_summary = aggregate_seed_rows(pd.DataFrame(correlation_rows), ["score_1", "score_2"], ["spearman_rho"])
    class_summary = aggregate_seed_rows(pd.DataFrame(class_rows), ["method", "subset", "label"], ["prevalence", "f1"])
    subgroup_summary = aggregate_seed_rows(pd.DataFrame(subgroup_rows), ["method", "group_type", "group"], ["n_records", "rejection_rate", "macro_f1_all", "sample_error_all"])
    ece_summary = aggregate_seed_rows(pd.DataFrame(ece_rows), ["label"], ["raw_event_probability_ece", "calibrated_event_probability_ece"])
    reliability_summary = aggregate_seed_rows(pd.DataFrame(reliability_rows), ["subset"], ["n_records", "sample_f1_loss", "exact_match_error", "brier", "nll", "macro_f1"])

    targets = {
        "Macro-AUC": (0.8971, 0.0018, baseline["macro_auc"].to_numpy()),
        "Macro-F1": (0.6864, 0.0061, baseline["macro_f1"].to_numpy()),
        "Micro-F1": (0.7334, 0.0039, baseline["micro_f1"].to_numpy()),
    }
    paper_rows = seed_metrics[seed_metrics["method"] == "RCUF-3 (paper)"]
    paper_calibration = pd.DataFrame(paper_calibration_rows)
    targets.update(
        {
            "RCUF Macro-F1@90": (0.7101, 0.0062, paper_rows["macro_f1_90"].to_numpy()),
            "RCUF Risk@90": (0.2372, 0.0033, paper_rows["risk_90"].to_numpy()),
            "RCUF pAURC (coarse)": (0.1026, 0.0017, paper_rows["coarse_paurc"].to_numpy()),
            "RCUF rejected/accepted ratio": (2.2357, 0.2540, paper_rows["error_ratio_90"].to_numpy()),
            "Raw Micro-ECE": (0.0474, 0.0098, paper_calibration["raw_micro_ece"].to_numpy()),
            "Temperature-scaled Micro-ECE": (0.0311, 0.0053, paper_calibration["calibrated_micro_ece"].to_numpy()),
        }
    )
    reproduction_rows = []
    for metric, (paper_mean, paper_sd, values) in targets.items():
        recalculated_mean, recalculated_sd = mean_sd(values)
        reproduction_rows.append(
            {
                "metric": metric,
                "paper_mean": paper_mean,
                "recalculated_mean": recalculated_mean,
                "paper_sd": paper_sd,
                "recalculated_sd": recalculated_sd,
                "matches_rounding": f"{paper_mean:.4f}" == f"{recalculated_mean:.4f}" and f"{paper_sd:.4f}" == f"{recalculated_sd:.4f}",
            }
        )
    reproduction = pd.DataFrame(reproduction_rows)
    bootstrap = bootstrap_differences(seed_payloads, args.bootstrap, args.bootstrap_seed)

    outputs = {
        "reproduction_audit.csv": reproduction,
        "selective_results_by_seed.csv": seed_metrics,
        "selective_results_summary.csv": summary,
        "dense_risk_coverage_source_data.csv": pd.concat(curve_rows, ignore_index=True),
        "calibration_development_results.csv": calibration_metrics,
        "calibration_development_summary.csv": calibration_summary,
        "calibration_threshold_transfer_by_seed.csv": transfers,
        "calibration_threshold_transfer_summary.csv": transfer_summary,
        "score_correlations_by_seed.csv": pd.DataFrame(correlation_rows),
        "score_correlations_summary.csv": correlation_summary,
        "accepted_rejected_per_class_by_seed.csv": pd.DataFrame(class_rows),
        "accepted_rejected_per_class_summary.csv": class_summary,
        "subgroup_rejection_by_seed.csv": pd.DataFrame(subgroup_rows),
        "subgroup_rejection_summary.csv": subgroup_summary,
        "classwise_event_ece_by_seed.csv": pd.DataFrame(ece_rows),
        "classwise_event_ece_summary.csv": ece_summary,
        "paper_micro_ece_by_seed.csv": paper_calibration,
        "accepted_rejected_reliability_by_seed.csv": pd.DataFrame(reliability_rows),
        "accepted_rejected_reliability_summary.csv": reliability_summary,
        "paired_bootstrap_differences.csv": bootstrap,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    write_report(output_dir, reproduction, summary, calibration_summary, transfer_summary, bootstrap, ece_summary, subgroup_summary)
    print(f"Revision analyses written to {output_dir}")


if __name__ == "__main__":
    main()
