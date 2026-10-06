import ast
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(scp_codes_str):
    if isinstance(scp_codes_str, dict):
        return scp_codes_str
    if pd.isna(scp_codes_str):
        return {}
    try:
        parsed = ast.literal_eval(str(scp_codes_str))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_scp_statements(path):
    df = pd.read_csv(path, index_col=0)
    if "diagnostic" not in df.columns:
        raise ValueError("scp_statements.csv missing diagnostic column")
    return df


def map_to_diagnostic_superclasses(row_scp_codes, scp_statements):
    labels = set()
    for code in row_scp_codes.keys():
        if code not in scp_statements.index:
            continue
        row = scp_statements.loc[code]
        diagnostic = row.get("diagnostic", 0)
        if pd.isna(diagnostic) or int(diagnostic) != 1:
            continue
        superclass = row.get("diagnostic_class")
        if isinstance(superclass, str) and superclass in LABEL_ORDER:
            labels.add(superclass)
    return labels


def build_multilabel_targets(ptbxl_database, scp_statements):
    y = np.zeros((len(ptbxl_database), len(LABEL_ORDER)), dtype=np.int64)
    label_sets = []
    for i, codes_raw in enumerate(ptbxl_database["scp_codes"]):
        codes = parse_scp_codes(codes_raw)
        labels = map_to_diagnostic_superclasses(codes, scp_statements)
        label_sets.append(sorted(labels))
        for lab in labels:
            y[i, LABEL_ORDER.index(lab)] = 1
    out = ptbxl_database.copy()
    for j, lab in enumerate(LABEL_ORDER):
        out[f"label_{lab}"] = y[:, j]
    out["diagnostic_superclasses"] = ["|".join(x) for x in label_sets]
    out["n_superclasses"] = y.sum(axis=1)
    return out, y
