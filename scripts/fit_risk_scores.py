import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiment_utils import LABEL_ORDER, ensure, make_loader, make_model, predict_with_embedding
from src.ptbxl.risk.prototype import build_label_prototypes, prototype_risk
from src.ptbxl.risk.risk_scores import compute_basic_risks, multilabel_sample_f1, rank_fusion


def load_thresholds(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    th = data["thresholds"]
    return np.array([float(th[l]) for l in LABEL_ORDER], dtype=float)


def export_embeddings(path, y, logits, prob, emb, meta):
    np.savez_compressed(
        path,
        embedding=emb.astype(np.float32),
        logits=logits.astype(np.float32),
        prob=prob.astype(np.float32),
        y_true=y.astype(np.int64),
        ecg_id=np.asarray(meta["ecg_id"]),
        patient_id=np.asarray(meta["patient_id"]),
        strat_fold=np.asarray(meta["strat_fold"]),
    )


def make_score_df(y, logits, prob, emb, meta, thresholds, prototypes, cal_ref=None):
    pred = (prob >= thresholds).astype(int)
    sample_f1 = multilabel_sample_f1(y, pred)
    exact_error = (pred != y).any(axis=1).astype(int)
    risks = compute_basic_risks(logits, prob)
    d_all, d_pred, _ = prototype_risk(emb, prototypes, LABEL_ORDER, pred=pred)
    df = pd.DataFrame(meta)
    for i, lab in enumerate(LABEL_ORDER):
        df[f"true_{lab}"] = y[:, i].astype(int)
    for i, lab in enumerate(LABEL_ORDER):
        df[f"prob_{lab}"] = prob[:, i]
    for i, lab in enumerate(LABEL_ORDER):
        df[f"pred_{lab}"] = pred[:, i].astype(int)
    df["sample_f1"] = sample_f1
    df["sample_error"] = 1.0 - sample_f1
    df["exact_match_error"] = exact_error
    for k, v in risks.items():
        df[k] = v
    df["Prototype_risk"] = d_all
    df["Prototype_D_pred"] = d_pred
    if cal_ref is None:
        cal_ref = df
    paper_risk_cols = ["MSP_risk", "Entropy_risk", "Energy_risk"]
    df["Risk_rank_fusion"] = rank_fusion(cal_ref, df, paper_risk_cols)
    df["RCUF_3"] = df["Risk_rank_fusion"]
    df["RCUF_4_prototype_ablation"] = rank_fusion(
        cal_ref, df, paper_risk_cols + ["Prototype_risk"]
    )
    return df


def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_aupr(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def diagnostics(df):
    out = {}
    for col in ["MSP_risk", "Entropy_risk", "Energy_risk", "Prototype_risk", "RCUF_3", "RCUF_4_prototype_ablation"]:
        rho = spearmanr(df[col].values, df["sample_error"].values, nan_policy="omit").correlation
        high = (df["sample_f1"].values < 1.0).astype(int)
        severe = (df["sample_f1"].values < 0.5).astype(int)
        out[col] = {
            "spearman_sample_error": None if np.isnan(rho) else float(rho),
            "high_error_f1_auroc": safe_auc(high, df[col].values),
            "high_error_f1_aupr": safe_aupr(high, df[col].values),
            "high_error_severe_auroc": safe_auc(severe, df[col].values),
            "high_error_severe_aupr": safe_aupr(severe, df[col].values),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--cal-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl")
    ap.add_argument("--prefix", default="gpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if args.prefix == "gpu" and "gpu" not in Path(args.checkpoint).name:
        raise RuntimeError("P3 GPU risk scores require the GPU checkpoint.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    batch_size = int(ckpt.get("train_config", {}).get("batch_size", 16))
    num_workers = int(ckpt.get("train_config", {}).get("num_workers", 2))
    prefer_wfdb = bool(ckpt.get("train_config", {}).get("prefer_wfdb_streaming", True))
    loaders = {
        "train": make_loader(args.train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=device.type == "cuda"),
        "cal": make_loader(args.cal_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=device.type == "cuda"),
        "test": make_loader(args.test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=device.type == "cuda"),
    }
    packs = {}
    for name, loader in loaders.items():
        packs[name] = predict_with_embedding(model, loader, device)
        y, logits, prob, emb, meta = packs[name]
        export_embeddings(out / f"embeddings/ptbxl_{name}_embeddings_{args.prefix}_seed{args.seed}.npz", y, logits, prob, emb, meta)
    ytr, _, _, etr, _ = packs["train"]
    prototypes, counts = build_label_prototypes(etr, ytr, LABEL_ORDER)
    np.savez_compressed(
        out / f"calibration/ptbxl_prototypes_{args.prefix}_seed{args.seed}.npz",
        prototypes=np.vstack([prototypes[l] for l in LABEL_ORDER]).astype(np.float32),
        label_names=np.asarray(LABEL_ORDER),
        counts=np.asarray([counts[l] for l in LABEL_ORDER]),
    )
    thresholds = load_thresholds(args.thresholds)
    yc, lc, pc, ec, mc = packs["cal"]
    cal_df = make_score_df(yc, lc, pc, ec, mc, thresholds, prototypes)
    yt, lt, pt, et, mt = packs["test"]
    test_df = make_score_df(yt, lt, pt, et, mt, thresholds, prototypes, cal_ref=cal_df)
    cal_df.to_csv(out / f"predictions/ptbxl_cal_risk_scores_{args.prefix}_seed{args.seed}.csv", index=False)
    test_df.to_csv(out / f"predictions/ptbxl_test_risk_scores_{args.prefix}_seed{args.seed}.csv", index=False)
    np.savez_compressed(
        out / f"calibration/ptbxl_risk_reference_{args.prefix}_seed{args.seed}.npz",
        MSP_risk=cal_df["MSP_risk"].values,
        Entropy_risk=cal_df["Entropy_risk"].values,
        Energy_risk=cal_df["Energy_risk"].values,
        Prototype_risk=cal_df["Prototype_risk"].values,
    )
    diag = {"cal": diagnostics(cal_df), "test": diagnostics(test_df), "prototype_counts": counts}
    diag["pass_checks"] = {
        "rank_fusion_in_0_1": bool(((cal_df["Risk_rank_fusion"] >= 0) & (cal_df["Risk_rank_fusion"] <= 1)).all() and ((test_df["Risk_rank_fusion"] >= 0) & (test_df["Risk_rank_fusion"] <= 1)).all()),
        "no_nan_inf": bool(np.isfinite(cal_df.select_dtypes(include=[np.number]).values).all() and np.isfinite(test_df.select_dtypes(include=[np.number]).values).all()),
        "positive_spearman": any((v["spearman_sample_error"] or 0) > 0 for v in diag["test"].values()),
        "auroc_gt_0p55": any((v["high_error_f1_auroc"] or 0) > 0.55 for v in diag["test"].values()),
    }
    status = "PASS" if all(diag["pass_checks"].values()) else "FAIL"
    diag["status"] = status
    (out / f"metrics/ptbxl_risk_diagnostics_{args.prefix}_seed{args.seed}.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    report = ["# PTB-XL P3 GPU Risk Score Report", "", f"PASS/FAIL: {status}", f"checkpoint: {args.checkpoint}", "Risk direction: larger score means higher rejection risk.", "", json.dumps(diag, indent=2)]
    (out / f"reports/phase_p3_{args.prefix}_risk_score_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "pass_checks": diag["pass_checks"]}, indent=2))


if __name__ == "__main__":
    main()
