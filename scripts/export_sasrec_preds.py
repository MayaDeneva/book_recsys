"""Export SASRec top-10 predictions per test user via RecBole's full_sort_topk.

Requires RecBole (``pip install recbole``) and the offline artifacts:

- ``artifacts/SASRec.pth``       — trained checkpoint from notebook 06_recbole.ipynb
- ``artifacts/sample.parquet``   — source interactions (for the N_USERS subsample)

This generates the **RAW (non-denoised)** ground-truth predictions matching the shipped
``SASRec.pth``, which was trained with ``DENOISE_HISTORY=False``: subsample N_USERS=30000
(seed 42) → cap to last MAX_HIST=100. NO shelf-denoising or user-set equalization is
applied, because the shipped checkpoint's item vocabulary / dataset is the raw one. A
future *denoised* checkpoint would need the denoised data path re-added here (mirroring
notebook 06 cell 4 with ``DENOISE_HISTORY=True``) to stay consistent with its training.

This script is **offline-only**: the live serving path (book_recsys package) does NOT
import RecBole and uses ``artifacts/SASRec_state.pt`` instead.  Run this script whenever
the checkpoint is retrained to regenerate the ground-truth predictions consumed by
``scripts/verify_sasrec_port.py``:

    python scripts/export_sasrec_preds.py

It rebuilds the RecBole dataset from scratch every run (save_dataset/save_dataloaders
OFF, and any stale ``saved/`` cache is deleted first) so the export is deterministic and
cache-independent — a fresh clone reproduces the verification gate exactly.

Output: ``artifacts/SASRec_preds.json`` — ``{user_token: [book_id, ...]}`` mapping,
top-10 per test user, in RecBole's internal evaluation order.

NOT under package coverage (scripts/ is excluded from coverage.ini).
"""
import functools
import json
import os

import numpy as np
import torch

# RecBole's compat shim references NumPy 2.0-removed aliases on import; patch before
# importing recbole so it does NOT require a numpy downgrade.
for _old, _new in [("float_", "float64"), ("complex_", "complex128"), ("unicode_", "str_")]:
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, init_logger, init_seed
from recbole.utils.case_study import full_sort_topk

from book_recsys.data.schema import USER
from book_recsys.recbole_adapter.atomic import write_inter_file
from book_recsys.recbole_adapter.export import recbole_predictions

# --- Constants (must match notebook 06 cell 4 and the trained checkpoint) ---
N_USERS = 30000
MAX_HIST = 100
DATASET = "goodreads"
MODEL = "SASRec"
USERS_PER_BATCH = 512
TOPK = 10

CHECKPOINT = "artifacts/SASRec.pth"
INTER_DIR = f"recbole_data/{DATASET}"
OUT_PATH = f"artifacts/{MODEL}_preds.json"


def load_sample() -> pd.DataFrame:
    """Load sample.parquet, subsample N_USERS with seed 42, cap to MAX_HIST.

    RAW data path (DENOISE_HISTORY=False): mirrors notebook 06 cell 4 *without* the
    shelf-denoising or user-set equalization, matching the shipped raw SASRec.pth. A
    future denoised checkpoint would need those steps re-added here.
    """
    sample = pd.read_parquet("artifacts/sample.parquet")
    print(f"loaded {sample[USER].nunique():,} users, {len(sample):,} interactions")

    keep = sample[USER].drop_duplicates().sample(N_USERS, random_state=42)
    sample = sample[sample[USER].isin(keep)]
    sample = (
        sample.sort_values([USER, "timestamp"])
        .groupby(USER, sort=False)
        .tail(MAX_HIST)
        .reset_index(drop=True)
    )
    print(f"after subsample + history cap: {sample[USER].nunique():,} users, "
          f"{len(sample):,} interactions")
    return sample


def build_recbole_config(device: str) -> Config:
    """Build RecBole Config mirroring notebook 06 cell 6."""
    return Config(
        model=MODEL,
        dataset=DATASET,
        config_dict={
            "data_path": "recbole_data",
            "dataset": DATASET,
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "TIME_FIELD": "timestamp",
            "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "MAX_ITEM_LIST_LENGTH": 50,
            "train_neg_sample_args": None,
            "epochs": 50,
            "topk": TOPK,
            "train_batch_size": 512,
            "eval_batch_size": 256,
            "eval_step": 5,
            "stopping_step": 3,
            "metrics": ["NDCG", "Recall", "MRR"],
            "valid_metric": "NDCG@10",
            # OFF so create_dataset always rebuilds from the freshly written .inter
            # rather than loading a stale cache (which would shadow the raw rebuild).
            "save_dataset": False,
            "save_dataloaders": False,
            "checkpoint_dir": "saved",
            "show_progress": False,
            "seed": 42,
            "reproducibility": True,
            "device": device,
        },
    )


def main() -> None:
    device = "cpu"  # export runs on CPU; checkpoint was trained on GPU

    # --- 0. Delete any stale RecBole cache so create_dataset rebuilds from the fresh
    # .inter. A leftover cache would silently shadow the rebuild and defeat reproducibility.
    for stale in (
        f"saved/{DATASET}-SequentialDataset.pth",
        f"saved/{DATASET}-for-{MODEL}-dataloader.pth",
    ):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"removed stale cache: {stale}")

    # --- 1. Rebuild the .inter file from sample.parquet (same as notebook cell 4) ---
    sample = load_sample()
    os.makedirs(INTER_DIR, exist_ok=True)
    write_inter_file(sample, f"{INTER_DIR}/{DATASET}.inter")
    print(f"wrote {INTER_DIR}/{DATASET}.inter")

    # --- 2. Build RecBole config + dataset + dataloaders (mirrors notebook cells 6-7) ---
    config = build_recbole_config(device)
    # Monkey-patch torch.load so RecBole's cached dataset/dataloader loads don't fail on
    # PyTorch >=2.6 (weights_only=True default) and correctly map GPU-saved tensors to CPU.
    _orig_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False, map_location=device)
    try:
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        dataset = create_dataset(config)
        train_data, _valid_data, test_data = data_preparation(config, dataset)

        # --- 3. Load checkpoint (mirrors notebook cell 8 / 10 alternative) ---
        ckpt = _orig_load(CHECKPOINT, map_location=device, weights_only=False)
        model = get_model(config["model"])(config, train_data.dataset).to(device)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("other_parameter"):
            model.load_other_parameter(ckpt["other_parameter"])
        model.eval()
        print(f"loaded {MODEL}: epoch {ckpt.get('epoch')}, best valid {ckpt.get('best_valid_score')}")

        # --- 4. full_sort_topk batched over ALL internal users (mirrors notebook cell 9) ---
        internal_users = list(range(1, dataset.user_num))  # skip [PAD]=0
        uid2token = {i: dataset.id2token(dataset.uid_field, i) for i in internal_users}

        chunks = []
        for s in range(0, len(internal_users), USERS_PER_BATCH):
            batch = internal_users[s: s + USERS_PER_BATCH]
            _, iid = full_sort_topk(batch, model, test_data, k=TOPK, device=device)
            chunks.append(iid.cpu())
        topk_iid = torch.cat(chunks, dim=0)

        iid2token = {
            int(i): dataset.id2token(dataset.iid_field, int(i))
            for i in topk_iid.reshape(-1).unique().tolist()
        }
        preds = recbole_predictions(internal_users, topk_iid.tolist(), uid2token, iid2token)
    finally:
        torch.load = _orig_load

    # --- 5. Write output ---
    os.makedirs("artifacts", exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(preds, fh)
    print(f"exported top-{TOPK} for {len(preds):,} users -> {OUT_PATH}")


if __name__ == "__main__":
    main()
