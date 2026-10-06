import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             precision_recall_fscore_support, roc_auc_score)


def compute_multilabel_metrics(y_true, y_prob, y_pred, label_names):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    per_auc, per_auprc = {}, {}
    available, skipped = [], []
    for i, name in enumerate(label_names):
        if len(np.unique(y_true[:, i])) == 2:
            per_auc[name] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            per_auprc[name] = float(average_precision_score(y_true[:, i], y_prob[:, i]))
            available.append(name)
        else:
            per_auc[name] = None
            per_auprc[name] = None
            skipped.append(name)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    out = {
        "macro_auc": float(np.mean([v for v in per_auc.values() if v is not None])) if available else None,
        "micro_auc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())) if len(np.unique(y_true.ravel())) == 2 else None,
        "macro_auprc": float(np.mean([v for v in per_auprc.values() if v is not None])) if available else None,
        "micro_auprc": float(average_precision_score(y_true.ravel(), y_prob.ravel())),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class_auc": per_auc,
        "per_class_auprc": per_auprc,
        "per_class_f1": {name: float(f[i]) for i, name in enumerate(label_names)},
        "per_class_precision": {name: float(p[i]) for i, name in enumerate(label_names)},
        "per_class_recall": {name: float(r[i]) for i, name in enumerate(label_names)},
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "available_auc_labels": available,
        "skipped_auc_labels": skipped,
    }
    return out
