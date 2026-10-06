import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ptbxl.data.labels import LABEL_ORDER, build_multilabel_targets, load_scp_statements


def ensure_dirs(out_dir):
    out = Path(out_dir)
    for sub in ["data", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def robust_norm(x):
    med = np.median(x, axis=0, keepdims=True)
    q75 = np.percentile(x, 75, axis=0, keepdims=True)
    q25 = np.percentile(x, 25, axis=0, keepdims=True)
    return (x - med) / (q75 - q25 + 1e-6)


def fix_length(sig, target=1000):
    n = sig.shape[0]
    if n == target:
        return sig, "none"
    if n < target:
        pad = np.zeros((target - n, sig.shape[1]), dtype=sig.dtype)
        return np.vstack([sig, pad]), "pad"
    start = (n - target) // 2
    return sig[start:start + target], "crop"


def load_signal(data_root, filename):
    sig, meta = wfdb.rdsamp(str(data_root / filename))
    sig, adj = fix_length(sig.astype(np.float32), 1000)
    sig = robust_norm(sig).astype(np.float32)
    return sig.T, adj


def save_split(df, y, data_root, name, out):
    xs, adjs = [], []
    for i, row in df.iterrows():
        x, adj = load_signal(data_root, row["filename_lr"])
        xs.append(x)
        adjs.append(adj)
        if len(xs) % 1000 == 0:
            print(f"{name}: loaded {len(xs)}/{len(df)}")
    X = np.stack(xs).astype(np.float32)
    yy = y[df.index.to_numpy()].astype(np.float32)
    arrays = {
        "X": X,
        "y": yy,
        "ecg_id": df["ecg_id"].to_numpy(),
        "patient_id": df["patient_id"].to_numpy() if "patient_id" in df.columns else np.array([""] * len(df)),
        "strat_fold": df["strat_fold"].to_numpy(),
        "filename": df["filename_lr"].astype(str).to_numpy(),
        "age": df["age"].to_numpy() if "age" in df.columns else np.full(len(df), np.nan),
        "sex": df["sex"].to_numpy() if "sex" in df.columns else np.array([""] * len(df)),
        "length_adjustment": np.array(adjs),
    }
    np.savez_compressed(out / f"data/ptbxl_{name}_100hz.npz", **arrays)
    return X.shape, yy.shape, {a: int(adjs.count(a)) for a in sorted(set(adjs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl")
    ap.add_argument("--sampling-rate", type=int, default=100)
    ap.add_argument("--split-dir", default="splits")
    args = ap.parse_args()
    out = ensure_dirs(args.out_dir)
    root = Path(args.data_root)
    db = pd.read_csv(root / "ptbxl_database.csv")
    scp = load_scp_statements(root / "scp_statements.csv")
    labeled, y = build_multilabel_targets(db, scp)
    if args.sampling_rate != 100:
        raise SystemExit("FAIL: this P1 implementation expects records100 / 100Hz")
    if "filename_lr" not in labeled.columns:
        raise SystemExit("FAIL: filename_lr missing")
    train = labeled[labeled.strat_fold.isin(range(1, 9))].copy()
    fold9 = labeled[labeled.strat_fold == 9].copy().sort_values("ecg_id")
    val = fold9[fold9.ecg_id.astype(int) % 2 == 0].copy()
    cal = fold9[fold9.ecg_id.astype(int) % 2 == 1].copy()
    test = labeled[labeled.strat_fold == 10].copy()
    shapes = {}
    adj = {}
    for name, df in [("train", train), ("val", val), ("cal", cal), ("test", test)]:
        shapes[name], _, adj[name] = save_split(df, y, root, name, out)
    meta = labeled[["ecg_id", "patient_id", "strat_fold", "filename_lr", "age", "sex"] + [f"label_{l}" for l in LABEL_ORDER]].copy()
    meta["split"] = np.select(
        [meta.strat_fold.isin(range(1, 9)), (meta.strat_fold == 9) & (meta.ecg_id.astype(int) % 2 == 0), (meta.strat_fold == 9) & (meta.ecg_id.astype(int) % 2 == 1), meta.strat_fold == 10],
        ["train", "val", "cal", "test"],
        default="unused",
    )
    meta.to_csv(out / "data/ptbxl_metadata_processed.csv", index=False)
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("validation", val), ("calibration", cal), ("test", test)]:
        frame[["ecg_id"]].to_csv(split_dir / f"{name}_ids.csv", index=False)
    nan_ok = True
    for name in ["train", "val", "cal", "test"]:
        d = np.load(out / f"data/ptbxl_{name}_100hz.npz")
        nan_ok = nan_ok and np.isfinite(d["X"]).all() and np.isfinite(d["y"]).all()
    report = [
        "# PTB-XL Phase P1 Preprocessing Report",
        "",
        "Status: PASS" if nan_ok else "Status: FAIL",
        f"split_counts: train={len(train)}, val={len(val)}, cal={len(cal)}, test={len(test)}",
        f"shapes: {shapes}",
        f"length_adjustments: {adj}",
        "train folds = 1-8; val/cal fold = 9 split by ecg_id parity; test fold = 10",
        f"nan_inf_check: {'PASS' if nan_ok else 'FAIL'}",
    ]
    (out / "reports/phase_p1_preprocessing_report.md").write_text("\n".join(report), encoding="utf-8")
    print({"status": "PASS" if nan_ok else "FAIL", "shapes": shapes})
    if not nan_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
