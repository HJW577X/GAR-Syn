import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

from .constants import ID_COLUMNS, LABELS, RAW_FILES


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def drop_unnamed(df):
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")


def check_columns(df, labels):
    missing = [c for c in ID_COLUMNS + labels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_random(data, repeat):
    train, temp = train_test_split(data, test_size=0.2, random_state=repeat, shuffle=True)
    val, test = train_test_split(temp, test_size=0.5, random_state=repeat, shuffle=True)
    return train, val, test


def split_drugout(data, repeat):
    np.random.seed(repeat)
    drugs = np.unique(np.concatenate([data["drug_row"].values, data["drug_col"].values]))
    np.random.shuffle(drugs)
    train_drugs = set(drugs[: int(len(drugs) * 0.8)])
    heldout_drugs = set(drugs[int(len(drugs) * 0.8) :])
    train = data[data["drug_row"].isin(train_drugs) & data["drug_col"].isin(train_drugs)]
    val_test = data[data["drug_row"].isin(heldout_drugs) & data["drug_col"].isin(heldout_drugs)]
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)
    return train, val, test


def split_drugpairout(data, repeat):
    pairs = data[["drug_row", "drug_col"]].apply(lambda x: tuple(sorted(x)), axis=1)
    unique_pairs = pairs.unique()
    np.random.seed(repeat)
    np.random.shuffle(unique_pairs)
    train_pairs = set(unique_pairs[: int(len(unique_pairs) * 0.8)])
    heldout_pairs = set(unique_pairs[int(len(unique_pairs) * 0.8) :])
    train = data[pairs.isin(train_pairs)]
    val_test = data[pairs.isin(heldout_pairs)]
    val, test = train_test_split(val_test, test_size=0.5, random_state=repeat)
    return train, val, test


def split_cellout(data, repeat):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=repeat)
    train_idx, valtest_idx = next(gss.split(data, groups=data["depmap"]))
    train = data.iloc[train_idx]
    valtest = data.iloc[valtest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=repeat)
    val_idx, test_idx = next(gss2.split(valtest, groups=valtest["depmap"]))
    return train, valtest.iloc[val_idx], valtest.iloc[test_idx]


def split_bothout(data, repeat):
    cells = pd.unique(data["depmap"].values.ravel())
    pairs = data[["drug_row", "drug_col"]].apply(lambda x: tuple(sorted(x)), axis=1).unique()
    np.random.seed(repeat)
    np.random.shuffle(cells)
    np.random.shuffle(pairs)
    cell_rank = {cell: idx for idx, cell in enumerate(cells)}
    train_pairs = set(pairs[: int(len(pairs) * 0.8)])
    heldout_pairs = set(pairs[int(len(pairs) * 0.8) :])
    train_cell_size = int(len(cells) * 0.8)
    pair_series = data[["drug_row", "drug_col"]].apply(lambda x: tuple(sorted(x)), axis=1)
    train = data[pair_series.isin(train_pairs) & (data["depmap"].map(cell_rank) < train_cell_size)]
    val_test = data[pair_series.isin(heldout_pairs) & (data["depmap"].map(cell_rank) >= train_cell_size)]
    val, test = train_test_split(val_test, test_size=0.5, random_state=repeat)
    return train, val, test


SPLITTERS = {
    "random": split_random,
    "drugout": split_drugout,
    "drugpairout": split_drugpairout,
    "cellout": split_cellout,
    "bothout": split_bothout,
}


DATASET_SOURCE_DIRS = {
    "drugcomb": None,
    "oneil": None,
    "nci-almanac": None,
}


def prepare_data(
    output_dir,
    dataset="drugcomb",
    source_raw_dir=None,
    source_repeat_dir=None,
    split_mode="random",
    folds=(1, 2, 3, 4, 5),
    labels=LABELS,
):
    output_dir = Path(output_dir)
    folds_dir = output_dir / "folds"
    raw_out = output_dir / "rawData" / "drugcomb"
    ensure_dir(folds_dir)
    ensure_dir(raw_out)

    if source_raw_dir is None:
        source_raw_dir = DATASET_SOURCE_DIRS[dataset]
    if source_raw_dir is None:
        raise ValueError(
            "source_raw_dir is required. Provide a directory containing the raw feature files "
            "and data_to_split.csv, or pass --split-mode precomputed with --source-repeat-dir."
        )
    source_raw_dir = Path(source_raw_dir)
    source_repeat_dir = Path(source_repeat_dir) if source_repeat_dir is not None else None

    raw_data = None
    if split_mode != "precomputed":
        raw_data = drop_unnamed(pd.read_csv(source_raw_dir / "data_to_split.csv"))
        check_columns(raw_data, labels)

    manifest = {
        "dataset": dataset,
        "split_mode": split_mode,
        "source_raw_dir": str(source_raw_dir),
        "source_repeat_dir": str(source_repeat_dir),
        "folds": list(folds),
        "labels": list(labels),
        "fold_files": [],
    }

    for fold in folds:
        if split_mode == "precomputed":
            if source_repeat_dir is None:
                raise ValueError("--source-repeat-dir is required when --split-mode precomputed")
            split_frames = {}
            for split in ["train", "val", "test"]:
                src = source_repeat_dir / f"repeat{fold}_{split}.csv"
                df = drop_unnamed(pd.read_csv(src))
                check_columns(df, labels)
                split_frames[split] = df
        else:
            train, val, test = SPLITTERS[split_mode](raw_data, fold)
            split_frames = {"train": train, "val": val, "test": test}

        for split, df in split_frames.items():
            out = folds_dir / f"repeat{fold}_{split}.csv"
            drop_unnamed(df).reset_index(drop=True).to_csv(out, index=False)
            manifest["fold_files"].append(str(out))

    for name in RAW_FILES:
        src = source_raw_dir / name
        dst = raw_out / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_feature_csv(path, drop_id=True):
    df = drop_unnamed(pd.read_csv(path))
    if drop_id and "id" in df.columns:
        df = df.drop(columns=["id"])
    return df.values.astype("float32")


class FeatureStore:
    def __init__(self, data_dir):
        raw = Path(data_dir) / "rawData" / "drugcomb"
        self.drug_fp = read_feature_csv(raw / "Drug_use.csv")
        seq = drop_unnamed(pd.read_csv(raw / "drug_sequence_em.csv"))
        self.drug_seq = seq.iloc[:, 1:769].values.astype("float32") if seq.shape[1] > 768 else seq.values.astype("float32")
        self.cell_exp = read_feature_csv(raw / "Cell_use_zscore.csv")
        self.cell_mu = read_feature_csv(raw / "mutation.csv", drop_id=True)
        self.cell_nv = read_feature_csv(raw / "nv_zscore.csv", drop_id=True)
        self.drug_graph = np.load(raw / "drug_feature_graph.npy", allow_pickle=True).item()
        self.drug_map = np.load(raw / "Drug_map.npy", allow_pickle=True).item()
        self.reverse_drug_map = {value: key for key, value in self.drug_map.items()}


class SynergyDataset(Dataset):
    def __init__(self, data_dir, fold, split, label="S_mean"):
        self.path = Path(data_dir) / "folds" / f"repeat{fold}_{split}.csv"
        self.df = drop_unnamed(pd.read_csv(self.path))
        check_columns(self.df, [label])
        self.inputs = torch.from_numpy(self.df[ID_COLUMNS].astype("int64").values)
        self.labels = torch.from_numpy(self.df[label].astype("float32").values)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def prediction_frame(data_dir, fold):
    df = drop_unnamed(pd.read_csv(Path(data_dir) / "folds" / f"repeat{fold}_test.csv"))
    return df[ID_COLUMNS + LABELS].copy()
