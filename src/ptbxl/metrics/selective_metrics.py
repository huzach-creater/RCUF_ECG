import numpy as np

from src.ptbxl.data.labels import LABEL_ORDER
from src.ptbxl.metrics.multilabel_metrics import compute_multilabel_metrics


def subset_arrays(df):
    y = df[[f"true_{l}" for l in LABEL_ORDER]].values.astype(int)
    p = df[[f"prob_{l}" for l in LABEL_ORDER]].values.astype(float)
    pred = df[[f"pred_{l}" for l in LABEL_ORDER]].values.astype(int)
    return y, p, pred


def selective_table(df, risk_col, coverages):
    rows = []
    n = len(df)
    ordered = df.sort_values(risk_col, ascending=True).reset_index(drop=True)
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        acc = ordered.iloc[:k].copy()
        rej = ordered.iloc[k:].copy()
        y, p, pred = subset_arrays(acc)
        metrics = compute_multilabel_metrics(y, p, pred, LABEL_ORDER)
        row = {
            "risk_score": risk_col,
            "coverage": float(cov),
            "accepted_count": int(len(acc)),
            "rejected_count": int(len(rej)),
            "accepted_macro_f1": metrics["macro_f1"],
            "accepted_micro_f1": metrics["micro_f1"],
            "accepted_macro_auc": metrics["macro_auc"],
            "accepted_micro_auc": metrics["micro_auc"],
            "accepted_macro_auprc": metrics["macro_auprc"],
            "accepted_sample_error_mean": float(acc["sample_error"].mean()),
            "accepted_exact_match_error_rate": float(acc["exact_match_error"].mean()),
            "rejected_sample_error_mean": float(rej["sample_error"].mean()) if len(rej) else None,
            "rejected_exact_match_error_rate": float(rej["exact_match_error"].mean()) if len(rej) else None,
            "available_auc_labels": ",".join(metrics["available_auc_labels"]),
            "skipped_auc_labels": ",".join(metrics["skipped_auc_labels"]),
        }
        rows.append(row)
    return rows


def summarize_selective(table_df):
    out = {}
    for risk, g in table_df.groupby("risk_score"):
        by_cov = {round(float(r.coverage), 2): r for r in g.itertuples()}
        covs = np.array(g["coverage"].values, dtype=float)
        risks = np.array(g["accepted_sample_error_mean"].values, dtype=float)
        order = np.argsort(covs)
        aurc = float(np.trapz(risks[order], covs[order]))
        r100 = by_cov.get(1.0)
        r95 = by_cov.get(0.95)
        r90 = by_cov.get(0.9)
        r80 = by_cov.get(0.8)
        ratio90 = None
        if r90 is not None and r90.rejected_sample_error_mean is not None and r90.accepted_sample_error_mean > 0:
            ratio90 = float(r90.rejected_sample_error_mean / r90.accepted_sample_error_mean)
        out[risk] = {
            "NoReject_MacroF1": None if r100 is None else float(r100.accepted_macro_f1),
            "NoReject_MicroF1": None if r100 is None else float(r100.accepted_micro_f1),
            "NoReject_Risk": None if r100 is None else float(r100.accepted_sample_error_mean),
            "MacroF1@95": None if r95 is None else float(r95.accepted_macro_f1),
            "MacroF1@90": None if r90 is None else float(r90.accepted_macro_f1),
            "MacroF1@80": None if r80 is None else float(r80.accepted_macro_f1),
            "Risk@95": None if r95 is None else float(r95.accepted_sample_error_mean),
            "Risk@90": None if r90 is None else float(r90.accepted_sample_error_mean),
            "Risk@80": None if r80 is None else float(r80.accepted_sample_error_mean),
            "AURC": aurc,
            "AcceptedError@90": None if r90 is None else float(r90.accepted_sample_error_mean),
            "RejectedError@90": None if r90 is None or r90.rejected_sample_error_mean is None else float(r90.rejected_sample_error_mean),
            "Rejected/Accepted Error Ratio@90": ratio90,
        }
    return out
