import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def _find_ptbxl_root():
    for p in [
        Path("public/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
        Path("public/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
        Path("public/ptbxl"),
        Path("data/ptbxl"),
    ]:
        if (p / "ptbxl_database.csv").exists():
            return p
    return None


def _robust_norm(x):
    med = np.median(x, axis=0, keepdims=True)
    q75 = np.percentile(x, 75, axis=0, keepdims=True)
    q25 = np.percentile(x, 25, axis=0, keepdims=True)
    return np.nan_to_num((x - med) / (q75 - q25 + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)


def _fix_len(sig, target=1000):
    if sig.shape[0] == target:
        return sig
    if sig.shape[0] < target:
        return np.vstack([sig, np.zeros((target - sig.shape[0], sig.shape[1]), dtype=sig.dtype)])
    start = (sig.shape[0] - target) // 2
    return sig[start:start + target]


class PTBXLDataset(Dataset):
    def __init__(self, npz_path, indices=None, prefer_wfdb=True):
        self.npz_path = str(npz_path)
        self.data = np.load(self.npz_path, allow_pickle=True)
        n = len(self.data["y"])
        self.indices = np.arange(n) if indices is None else np.asarray(indices)
        self.root = _find_ptbxl_root()
        self.prefer_wfdb = prefer_wfdb and self.root is not None and "filename" in self.data.files
        self.cache = {} if len(self.indices) <= 2048 else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = None
        state["cache"] = {} if self.cache is not None else None
        return state

    def _ensure_data(self):
        if self.data is None:
            self.data = np.load(self.npz_path, allow_pickle=True)

    def __len__(self):
        return len(self.indices)

    def _load_x(self, i):
        self._ensure_data()
        if self.prefer_wfdb:
            if self.cache is not None and int(i) in self.cache:
                return self.cache[int(i)]
            import wfdb
            sig, _ = wfdb.rdsamp(str(self.root / str(self.data["filename"][i])))
            sig = _robust_norm(_fix_len(sig.astype(np.float32), 1000)).T.astype(np.float32)
            if self.cache is not None:
                self.cache[int(i)] = sig
            return sig
        return self.data["X"][i]

    def __getitem__(self, idx):
        self._ensure_data()
        i = self.indices[idx]
        return {
            "x": torch.from_numpy(self._load_x(i)).float(),
            "y": torch.from_numpy(self.data["y"][i]).float(),
            "ecg_id": self.data["ecg_id"][i],
            "patient_id": self.data["patient_id"][i],
            "strat_fold": self.data["strat_fold"][i],
        }
