import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ptbxl.data.dataset import PTBXLDataset
from src.ptbxl.data.labels import LABEL_ORDER
from src.ptbxl.metrics.multilabel_metrics import compute_multilabel_metrics
from src.ptbxl.models.resnet1d import ResNet1D


def ensure(out_dir):
    out = Path(out_dir)
    for sub in ["checkpoints", "predictions", "metrics", "reports", "calibration", "embeddings", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_model(cfg):
    return ResNet1D(
        in_channels=int(cfg.get("input_channels", 12)),
        num_classes=int(cfg.get("num_classes", 5)),
        model_size=cfg.get("model_size", "tiny"),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        base_channels=int(cfg.get("base_channels", 16)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def pos_weight_from_y(y, clip=20.0):
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = neg / np.maximum(pos, 1)
    return torch.tensor(np.clip(w, 1.0, clip), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ys, meta = [], [], {"ecg_id": [], "patient_id": [], "strat_fold": []}
    for b in loader:
        x = b["x"].to(device)
        out = model(x)
        logits.append(out.cpu())
        ys.append(b["y"])
        for k in meta:
            meta[k].extend([str(v) for v in b[k]])
    logits = torch.cat(logits).numpy()
    y = torch.cat(ys).numpy().astype(int)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    prob = 1 / (1 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return y, logits, prob, meta


@torch.no_grad()
def predict_with_embedding(model, loader, device):
    model.eval()
    logits, emb, ys, meta = [], [], [], {"ecg_id": [], "patient_id": [], "strat_fold": []}
    for b in loader:
        x = b["x"].to(device, non_blocking=True)
        out, z = model(x, return_embedding=True)
        logits.append(out.cpu())
        emb.append(z.cpu())
        ys.append(b["y"])
        for k in meta:
            meta[k].extend([str(v) for v in b[k]])
    logits = torch.cat(logits).numpy()
    embedding = torch.cat(emb).numpy()
    y = torch.cat(ys).numpy().astype(int)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    prob = 1 / (1 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return y, logits, prob, embedding, meta


def train_epoch(model, loader, opt, loss_fn, device, grad_clip=5.0, accumulation_steps=1, amp=False):
    model.train()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))
    opt.zero_grad(set_to_none=True)
    accumulation_steps = max(1, int(accumulation_steps))
    for b in loader:
        step_idx = len(losses)
        x, y = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            loss = loss_fn(model(x), y)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        if (step_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if len(losses) % accumulation_steps != 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def make_loader(npz_path, batch_size=64, shuffle=False, indices=None, num_workers=0, prefer_wfdb=False, pin_memory=False):
    return DataLoader(
        PTBXLDataset(npz_path, indices=indices, prefer_wfdb=prefer_wfdb),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )


def thresholds_from_val(y, prob):
    out = {}
    from sklearn.metrics import f1_score
    for i, lab in enumerate(LABEL_ORDER):
        best = (0.0, 0.5)
        for t in np.linspace(0.05, 0.95, 91):
            f = f1_score(y[:, i], (prob[:, i] >= t).astype(int), zero_division=0)
            if f > best[0]:
                best = (float(f), float(t))
        out[lab] = best[1]
    return out


def write_pred_csv(path, meta, y, logits, prob, pred05, predopt):
    import pandas as pd
    df = pd.DataFrame(meta)
    for i, lab in enumerate(LABEL_ORDER):
        df[f"true_{lab}"] = y[:, i].astype(int)
        df[f"logit_{lab}"] = logits[:, i]
        df[f"prob_{lab}"] = prob[:, i]
        df[f"pred05_{lab}"] = pred05[:, i].astype(int)
        df[f"predopt_{lab}"] = predopt[:, i].astype(int)
    df.to_csv(path, index=False)
