import numpy as np

from src.ptbxl.data.labels import LABEL_ORDER


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def binary_ece(confidence, correct, n_bins=15):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1.0 else confidence < hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return ece


def prob_ece(prob, y, n_bins=15):
    prob = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob <= hi if hi == 1.0 else prob < hi)
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - prob[mask].mean()))
    return ece


def calibration_metrics(y, prob, thresholds, n_bins=15):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    pred = (prob >= thresholds).astype(int)
    eps = 1e-8
    nll = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
    brier_per = ((prob - y) ** 2).mean(axis=0)
    class_ece = {}
    for i, lab in enumerate(LABEL_ORDER):
        conf = np.maximum(prob[:, i], 1.0 - prob[:, i])
        correct = (pred[:, i] == y[:, i]).astype(int)
        class_ece[lab] = binary_ece(conf, correct, n_bins=n_bins)
    conf_all = np.maximum(prob, 1.0 - prob).ravel()
    correct_all = (pred == y).astype(int).ravel()
    return {
        "NLL": float(nll),
        "Brier": float(np.mean((prob - y) ** 2)),
        "Classwise_Brier": {lab: float(brier_per[i]) for i, lab in enumerate(LABEL_ORDER)},
        "Micro_ECE": binary_ece(conf_all, correct_all, n_bins=n_bins),
        "Prob_ECE": prob_ece(prob, y, n_bins=n_bins),
        "Classwise_ECE": class_ece,
        "Classwise_ECE_mean": float(np.mean(list(class_ece.values()))),
    }
