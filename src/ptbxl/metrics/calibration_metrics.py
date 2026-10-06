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


def decision_confidence(prob, pred):
    """Return confidence in the thresholded binary decision."""
    prob = np.asarray(prob, dtype=float)
    pred = np.asarray(pred, dtype=int)
    if prob.shape != pred.shape:
        raise ValueError("Probability and prediction arrays must have the same shape")
    return np.where(pred == 1, prob, 1.0 - prob)


def operational_decision_ece(y, prob, pred, n_bins=15):
    """ECE of thresholded decisions using confidence in the chosen class."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if y.shape != pred.shape:
        raise ValueError("Target and prediction arrays must have the same shape")
    confidence = decision_confidence(prob, pred)
    correct = (pred == y).astype(int)
    return binary_ece(confidence.ravel(), correct.ravel(), n_bins=n_bins)


def calibration_metrics(y, prob, thresholds=None, n_bins=15, predictions=None):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if predictions is None:
        if thresholds is None:
            raise ValueError("thresholds are required when predictions are not supplied")
        thresholds = np.asarray(thresholds, dtype=float)
        pred = (prob >= thresholds).astype(int)
    else:
        pred = np.asarray(predictions, dtype=int)
        if pred.shape != prob.shape:
            raise ValueError("Probability and prediction arrays must have the same shape")
    eps = 1e-8
    nll = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
    brier_per = ((prob - y) ** 2).mean(axis=0)
    class_ece = {}
    for i, lab in enumerate(LABEL_ORDER):
        class_ece[lab] = operational_decision_ece(
            y[:, i], prob[:, i], pred[:, i], n_bins=n_bins
        )
    return {
        "NLL": float(nll),
        "Brier": float(np.mean((prob - y) ** 2)),
        "Classwise_Brier": {lab: float(brier_per[i]) for i, lab in enumerate(LABEL_ORDER)},
        "Micro_ECE": operational_decision_ece(y, prob, pred, n_bins=n_bins),
        "Prob_ECE": prob_ece(prob, y, n_bins=n_bins),
        "Classwise_ECE": class_ece,
        "Classwise_ECE_mean": float(np.mean(list(class_ece.values()))),
    }
