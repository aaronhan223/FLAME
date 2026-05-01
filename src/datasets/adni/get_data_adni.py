"""ADNI multi-modal data pipeline (preprocessed branch only) for the
clinical-highmmt multitask MoE pipeline.

Ports the preprocessed-image branch of FlexMoE/data.py into the
clinical-highmmt encoder/dataloader contract used by mimic-iv and EMBED tasks.

Outputs per-batch tuples consumed by the ``diag`` branch of
src/train_structure_multitask_mimic.py and src/encoders.py:ADNIEncoder.
"""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


# Modality letter -> human-readable key used in output dicts
_MOD_KEYS = {"I": "image", "G": "genomic", "C": "clinical", "B": "biospecimen"}

# Module-level cache so we read each modality file at most once per process,
# even when --n_runs invokes data_prepare multiple times (one per seed).
_ADNI_CACHE: dict = {}


def _adni_path(args) -> str:
    return getattr(args, "adni_path", "/export/io79/data/schaud35/datasets/adni/adni_processed/")


def _image_csv(adni_dir: str) -> str:
    img_dir = os.path.join(adni_dir, "image")
    candidates = sorted(f for f in os.listdir(img_dir) if f.startswith("UCSFFSX7") and f.endswith(".csv"))
    if not candidates:
        raise FileNotFoundError(f"No UCSFFSX7*.csv found under {img_dir}")
    return os.path.join(img_dir, candidates[-1])


def _load_label_table(adni_dir: str):
    label_df = pd.read_csv(os.path.join(adni_dir, "label.csv"), index_col="PTID")
    if "Unnamed: 0" in label_df.columns:
        label_df = label_df.drop(columns=["Unnamed: 0"])
    label_df = label_df.dropna(subset=["DIAGNOSIS"]).copy()
    label_df["DIAGNOSIS"] = label_df["DIAGNOSIS"].astype(int) - 1
    return label_df


def _load_image_features(adni_dir: str, initial_filling: str):
    """Mirror the preprocessed-image branch of FlexMoE/data.py."""
    df = pd.read_csv(_image_csv(adni_dir))
    df["update_stamp"] = pd.to_datetime(df["update_stamp"], errors="coerce")
    idx = df.groupby("PTID")["update_stamp"].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    df.index = df["PTID"]
    feature_cols = [
        c for c in df.columns
        if c.startswith("ST") and (c.endswith("CV") or c.endswith("TA") or c.endswith("SV"))
    ]
    df = df[feature_cols]
    if initial_filling == "mean":
        df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)
    arr = df.apply(pd.to_numeric, errors="coerce").fillna(0).values
    arr = StandardScaler().fit_transform(arr)
    return df.index.tolist(), arr.astype(np.float32)


def _load_genomic(adni_dir: str, initial_filling: str):
    df = sc.read_h5ad(os.path.join(adni_dir, "genomic", "genomic_merged.h5ad")).to_df()
    if initial_filling == "mean":
        df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)
    arr = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df.values)
    return df.index.tolist(), arr.astype(np.float32)


def _load_clinical(adni_dir: str, initial_filling: str):
    suffix = "_mean.csv" if initial_filling == "mean" else ".csv"
    df = pd.read_csv(os.path.join(adni_dir, "clinical", f"clinical_merged{suffix}"), index_col=0)
    drop_cols = [c for c in df.columns if c.startswith(("PTCOGBEG", "PTADDX", "PTADBEG"))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.index.tolist(), df.values.astype(np.float32)


def _load_biospecimen(adni_dir: str, initial_filling: str):
    suffix = "_mean.csv" if initial_filling == "mean" else ".csv"
    df = pd.read_csv(os.path.join(adni_dir, "biospecimen", f"biospecimen_merged{suffix}"), index_col=0)
    return df.index.tolist(), df.values.astype(np.float32)


def _cache_key(adni_dir: str, initial_filling: str) -> str:
    return f"{adni_dir}|{initial_filling}"


def _load_all_modalities(args):
    """Load (and cache) the labelled PTID table plus per-modality feature matrices.

    Returns a dict keyed by ``image|genomic|clinical|biospecimen|label_df`` where
    each modality entry is ``{"index": list[ptid], "data": np.ndarray}``.
    """
    adni_dir = _adni_path(args)
    initial_filling = getattr(args, "initial_filling", "mean")
    cache_key = _cache_key(adni_dir, initial_filling)
    if cache_key in _ADNI_CACHE:
        return _ADNI_CACHE[cache_key]

    label_df = _load_label_table(adni_dir)
    img_idx, img_arr = _load_image_features(adni_dir, initial_filling)
    gen_idx, gen_arr = _load_genomic(adni_dir, initial_filling)
    clin_idx, clin_arr = _load_clinical(adni_dir, initial_filling)
    bio_idx, bio_arr = _load_biospecimen(adni_dir, initial_filling)

    bundle = {
        "label_df": label_df,
        "image": {"index": img_idx, "data": img_arr},
        "genomic": {"index": gen_idx, "data": gen_arr},
        "clinical": {"index": clin_idx, "data": clin_arr},
        "biospecimen": {"index": bio_idx, "data": bio_arr},
    }
    _ADNI_CACHE[cache_key] = bundle
    return bundle


def _eligible_ptids(bundle, mod_letters):
    """PTIDs with a label and present in every requested modality."""
    label_set = set(bundle["label_df"].index)
    eligible = label_set
    for letter in mod_letters:
        eligible = eligible & set(bundle[_MOD_KEYS[letter]]["index"])
    return sorted(eligible)


def _stratified_split(ptids, labels_arr, seed, ratios=(0.7, 0.15, 0.15)):
    """Deterministic stratified train/val/test split by class label.

    ``ptids`` is a list, ``labels_arr`` is the matching int label array. Returns
    three lists of PTIDs.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    rng = np.random.default_rng(int(seed))
    ptids = np.asarray(ptids)
    labels = np.asarray(labels_arr)
    train, val, test = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train.extend(ptids[cls_idx[:n_train]].tolist())
        val.extend(ptids[cls_idx[n_train:n_train + n_val]].tolist())
        test.extend(ptids[cls_idx[n_train + n_val:]].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _maybe_load_canonical_splits(adni_dir: str):
    """Use ``PTID_splits.json`` next to the data if present (canonical seed=42)."""
    path = os.path.join(adni_dir, "PTID_splits.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return (
        list(set(data["training"])),
        list(set(data["validation"])),
        list(set(data["testing"])),
    )


def get_splits_for_seed(args, ptids, labels_arr):
    """Return (train, val, test) PTID lists for the given seed.

    Convention (from the user): seed=42 should reproduce ``PTID_splits.json`` if
    it exists; otherwise we fall back to a stratified seeded split. All other
    seeds always use the seeded stratified split.
    """
    seed = int(getattr(args, "seed", 42))
    canonical = _maybe_load_canonical_splits(_adni_path(args))
    if seed == 42 and canonical is not None:
        train_set = set(canonical[0]) & set(ptids)
        val_set = set(canonical[1]) & set(ptids)
        test_set = set(canonical[2]) & set(ptids)
        if train_set and val_set and test_set:
            return sorted(train_set), sorted(val_set), sorted(test_set)
    return _stratified_split(ptids, labels_arr, seed)


class ADNIDataset(Dataset):
    """Yield per-PTID dict {idx, label, <modality letters>}.

    Each modality entry is a 1-D float32 numpy array (raw features). The
    encoder (``src.encoders.ADNIEncoder``) handles patch-embedding.
    """

    def __init__(self, args, mode, mod_letters, ptids, label_df, mod_data):
        self.args = args
        self.mode = mode
        self.mod_letters = list(mod_letters)
        self.ptids = list(ptids)
        self.label_df = label_df
        self.mod_data = mod_data  # {letter: {"id_to_row": dict, "data": ndarray}}

    def __len__(self):
        return len(self.ptids)

    def __getitem__(self, idx):
        ptid = self.ptids[idx]
        label = int(self.label_df.loc[ptid, "DIAGNOSIS"])
        out = {"idx": ptid, "label": label}
        for letter in self.mod_letters:
            row = self.mod_data[letter]["id_to_row"][ptid]
            out[letter] = self.mod_data[letter]["data"][row]
        return out


def _build_mod_data(bundle, ptids, mod_letters):
    mod_data = {}
    ptid_set = set(ptids)
    for letter in mod_letters:
        key = _MOD_KEYS[letter]
        index = bundle[key]["index"]
        data = bundle[key]["data"]
        # Restrict to eligible ptids, build {ptid: row} map
        id_to_row = {ptid: i for i, ptid in enumerate(index) if ptid in ptid_set}
        mod_data[letter] = {"id_to_row": id_to_row, "data": data}
    return mod_data


def _make_collate_fn(mod_letters):
    letters = list(mod_letters)

    def collate(batch):
        batch = [b for b in batch if b is not None]
        idx = [b["idx"] for b in batch]
        label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        mod_tensors = {
            letter: torch.tensor(np.stack([b[letter] for b in batch]), dtype=torch.float32)
            for letter in letters
        }
        return idx, label, mod_tensors

    return collate


def get_modality_input_dims(args):
    """Return ``{letter: int}`` feature-dim per modality, after preprocessing."""
    bundle = _load_all_modalities(args)
    return {
        "I": bundle["image"]["data"].shape[1],
        "G": bundle["genomic"]["data"].shape[1],
        "C": bundle["clinical"]["data"].shape[1],
        "B": bundle["biospecimen"]["data"].shape[1],
    }


def get_n_classes(args) -> int:
    bundle = _load_all_modalities(args)
    return int(bundle["label_df"]["DIAGNOSIS"].nunique())


def data_prepare(args, task="diag", modeltype=None, data=None):
    """Build (train, val, test) DataLoaders for the ADNI diagnosis task.

    Returns the same arity as the EMBED pipeline:
    ``(train_dl, val_dl, test_dl, train_ds, val_ds, test_ds)``.
    """
    del task, data  # unused (single ADNI task)
    bundle = _load_all_modalities(args)

    if modeltype:
        mod_letters = [m.strip() for m in modeltype.split("_") if m.strip()]
    else:
        mod_letters = [m.strip() for m in args.diag_mod.split("-") if m.strip()]
    mod_letters = [m for m in mod_letters if m in _MOD_KEYS]
    if not mod_letters:
        raise ValueError(f"No valid ADNI modality letters in modeltype/diag_mod: {modeltype} / {args.diag_mod}")

    eligible = _eligible_ptids(bundle, mod_letters)
    if len(eligible) == 0:
        raise RuntimeError(
            f"No PTIDs survive intersection of label + modalities {mod_letters} "
            f"under {_adni_path(args)}"
        )
    label_df = bundle["label_df"]
    labels_arr = label_df.loc[eligible, "DIAGNOSIS"].astype(int).values

    train_ids, val_ids, test_ids = get_splits_for_seed(args, eligible, labels_arr)
    print(
        f"[ADNI/diag seed={args.seed} mods={mod_letters}] "
        f"eligible={len(eligible)} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )

    mod_data = _build_mod_data(bundle, eligible, mod_letters)

    train_ds = ADNIDataset(args, "train", mod_letters, train_ids, label_df, mod_data)
    val_ds = ADNIDataset(args, "val", mod_letters, val_ids, label_df, mod_data)
    test_ds = ADNIDataset(args, "test", mod_letters, test_ids, label_df, mod_data)

    bs = getattr(args, "train_bs_adni", None) or getattr(args, "train_bs_embed", 8)
    eval_bs = getattr(args, "eval_batch_size", bs)
    collate = _make_collate_fn(mod_letters)

    train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=bs, collate_fn=collate)
    val_dl = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=eval_bs, collate_fn=collate)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=eval_bs, collate_fn=collate)

    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
