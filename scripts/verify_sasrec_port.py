"""Verify the standalone SASRec port reproduces RecBole's exported predictions.

For a sample of users present in artifacts/SASRec_preds.json, rebuild the test-time input
sequence from artifacts/sample.parquet (sorted by timestamp, capped to the last MAX_HIST,
final item held out), run the port, and compare top-10 to RecBole's. A faithful port
yields high overlap; mean overlap@10 >= 0.9 is the pass bar.

RecBole's full_sort_topk for sequential models scores all items and masks only [PAD]=0;
it does NOT suppress history items. The comparison here mirrors that behaviour so that
it is the forward pass faithfulness — not a serving-side seen-item filter difference —
that is being tested.

    python scripts/verify_sasrec_port.py
"""
import json

import numpy as np
import pandas as pd

from book_recsys.data.schema import BOOK, TS, USER
from book_recsys.models.sequential.recommender import SasRecRecommender

MAX_HIST = 100  # matches notebook 06 cell 4
N_SAMPLE = 300
N_USERS = 30000  # matches notebook 06 cell 4 subsample


def main() -> None:
    rec = SasRecRecommender.from_checkpoint("artifacts/SASRec_state.pt",
                                            "artifacts/SASRec_item_map.json", device="cpu")
    preds = json.load(open("artifacts/SASRec_preds.json"))
    df = pd.read_parquet("artifacts/sample.parquet")
    df[BOOK] = df[BOOK].astype(str)
    df[USER] = df[USER].astype(str)
    # Replicate notebook 06 cell 4: subsample N_USERS with the same seed, cap to MAX_HIST
    keep = df[USER].drop_duplicates().sample(N_USERS, random_state=42)
    df = df[df[USER].isin(keep)]
    df = df.sort_values([USER, TS]).groupby(USER, sort=False).tail(MAX_HIST).reset_index(
        drop=True)
    by_user = {u: g for u, g in df.sort_values([USER, TS]).groupby(USER, sort=False)}

    users = [u for u in preds if u in by_user][:N_SAMPLE]
    overlaps, top1 = [], 0
    for u in users:
        hist = list(by_user[u][BOOK])[:-1]  # drop held-out test target (last item)
        if not hist:
            continue
        # Mirror RecBole's full_sort_topk: score all items, mask PAD only (no history filter)
        scores = rec._scores(hist)
        scores[0] = float("-inf")  # PAD token
        got = [rec._tokens[int(i)] for i in np.argsort(-scores)[:10]]
        want = [str(b) for b in preds[u]]
        if not want:
            continue
        overlaps.append(len(set(got) & set(want)) / len(want))
        top1 += int(bool(got) and got[0] == want[0])
    n = len(overlaps)
    print(f"users compared: {n}")
    print(f"mean overlap@10: {sum(overlaps) / n:.3f}")
    print(f"exact top-1 match rate: {top1 / n:.3f}")
    print("PASS" if sum(overlaps) / n >= 0.9 else "FAIL — port likely diverges from RecBole")


if __name__ == "__main__":
    main()
