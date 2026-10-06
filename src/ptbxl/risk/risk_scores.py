import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def multilabel_sample_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = np.zeros(len(y_true), dtype=float)
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        tp = np.logical_and(yt == 1, yp == 1).sum()
        fp = np.logical_and(yt == 0, yp == 1).sum()
        fn = np.logical_and(yt == 1, yp == 0).sum()
        if tp + fp + fn == 0:
            out[i] = 1.0
        elif tp == 0:
            out[i] = 0.0
        else:
            out[i] = 2.0 * tp / (2.0 * tp + fp + fn)
    return out


def compute_basic_risks(logits, prob):
    eps = 1e-8
    confidence = np.maximum(prob, 1.0 - prob)
    msp_mean = 1.0 - confidence.mean(axis=1)
    msp_max = 1.0 - confidence.min(axis=1)
    entropy = -(prob * np.log(prob + eps) + (1.0 - prob) * np.log(1.0 - prob + eps)).mean(axis=1)
    energy = -np.abs(logits).mean(axis=1)
    return {
        "MSP_risk": msp_mean,
        "MSP_risk_max": msp_max,
        "Entropy_risk": entropy,
        "Energy_risk": energy,
    }


def empirical_rank(cal_scores, scores):
    cal_scores = np.asarray(cal_scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    return (np.searchsorted(np.sort(cal_scores), scores, side="right") + 1.0) / (len(cal_scores) + 1.0)


def rank_fusion(cal_df, df, risk_cols):
    ranks = []
    for col in risk_cols:
        ranks.append(empirical_rank(cal_df[col].values, df[col].values))
    return np.vstack(ranks).mean(axis=0)
