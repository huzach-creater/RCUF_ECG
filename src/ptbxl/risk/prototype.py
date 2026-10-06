import numpy as np


def build_label_prototypes(embedding, y, label_names):
    prototypes = {}
    counts = {}
    for i, lab in enumerate(label_names):
        mask = y[:, i].astype(bool)
        counts[lab] = int(mask.sum())
        if mask.any():
            prototypes[lab] = embedding[mask].mean(axis=0)
        else:
            prototypes[lab] = np.zeros(embedding.shape[1], dtype=np.float32)
    return prototypes, counts


def prototype_distances(embedding, prototypes, label_names):
    proto_mat = np.vstack([prototypes[lab] for lab in label_names]).astype(np.float32)
    diff = embedding[:, None, :] - proto_mat[None, :, :]
    return np.square(diff).sum(axis=2)


def prototype_risk(embedding, prototypes, label_names, pred=None):
    dist = prototype_distances(embedding, prototypes, label_names)
    d_all = dist.min(axis=1)
    if pred is None:
        return d_all, d_all, dist
    d_pred = np.zeros(len(embedding), dtype=float)
    for i in range(len(embedding)):
        pos = np.where(pred[i].astype(bool))[0]
        d_pred[i] = dist[i, pos].min() if len(pos) else d_all[i]
    return d_all, d_pred, dist
