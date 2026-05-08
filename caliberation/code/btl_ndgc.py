"""BTL ranking + NDCG evaluation for Qwen2.5 combat data.

Pipeline
--------
1.  Read pairwise combat outcomes from ``data/qwen_2.5_combat_data.csv``.
    Each row is (question_i, question_j, winner, style) where
    winner = 1 if Q_i wins, 0 if Q_j wins, -1 on parse failure.
2.  For each of the 7 styles, fit a Bradley-Terry-Luce model by MLE:

        P(i beats j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))

    via Zermelo's iterative algorithm (a.k.a. Minorization-Maximization).
3.  Rank the 100 questions by descending BTL strength (rank 1 = strongest =
    the model is most confident on it under that style).
4.  Join with ``data/qwen2.5results.csv_judged_results.csv`` to attach the
    per-question/per-style answer text and the binary judge result
    (``yes`` -> 1, ``no`` -> 0).
5.  Write ``data/qwen_btl_score.csv`` with columns
    ``question, style, btl_rank, answer, result``, sorted by (style, btl_rank).
6.  Compute NDCG@k for k in {1..N} per style on the BTL ordering with binary
    relevance = ``result``. Print and save the 7 NDCG@N numbers (one per
    style) to ``data/qwen_btl_ndcg.csv``.

Outputs
-------
- ``data/qwen_btl_score.csv`` (700 rows: 100 questions x 7 styles)
- ``data/qwen_btl_ndcg.csv``  (7 rows: one NDCG@N per style)
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


# Config

DATA_DIR     = "."
COMBAT_FILE  = os.path.join(DATA_DIR, "qwen_2.5_combat_data.csv")
JUDGED_FILE  = os.path.join(DATA_DIR, "qwen2.5results_judged_results.csv")
OUT_RANK     = os.path.join(DATA_DIR, "qwen2.5_btl_score.csv")
OUT_NDCG     = os.path.join(DATA_DIR, "qwen2.5_btl_ndcg.csv")

STYLES = [
    "base",
    "confident_1",
    "confident_2",
    "doubtful_1",
    "doubtful_2",
    "evidential_1",
    "evidential_2",
]

# BTL MLE iteration controls
MAX_ITER = 10000
TOL      = 1e-9



# IO helpers

def load_combat(path: str):
    """Return {style: [(q_i, q_j, winner), ...]} dropping winner == -1."""
    out: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                w = int(row["winner"])
            except ValueError:
                continue
            if w not in (0, 1):
                continue
            out[row["style"]].append((row["question_i"], row["question_j"], w))
    return out


def load_judged(path: str, n: int):
    """Return list of dicts for the top-n rows of the judged file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    return rows



# BTL via Zermelo / MM

def fit_btl(items: List[str], pairs: List[Tuple[str, str, int]]):
    """Maximum-likelihood Bradley-Terry-Luce.

    Uses the classical MM (Zermelo) update on strengths p_i > 0:

        p_i  <-  W_i / sum_{j != i} N_{ij} / (p_i + p_j)

    where W_i is the total wins of i across all opponents and N_{ij} is the
    total number of contests between i and j. Iterates to a fixed point and
    normalizes so that sum(p) == 1. Returns ``beta = log(p)`` so that the
    BTL probability is the standard logistic on ``beta_i - beta_j``.

    Disconnected items (no comparisons) get strength 0 / -inf and are ranked
    last; in our setup R = 10 random samples per question, so the comparison
    graph is dense and connected with overwhelming probability.
    """
    n = len(items)
    idx = {q: k for k, q in enumerate(items)}

    # Wins per item, and pair counts N[i][j] = N[j][i]
    W = [0.0] * n
    N: List[Dict[int, float]] = [defaultdict(float) for _ in range(n)]
    for qi, qj, w in pairs:
        if qi not in idx or qj not in idx:
            continue
        i, j = idx[qi], idx[qj]
        if i == j:
            continue
        N[i][j] += 1.0
        N[j][i] += 1.0
        if w == 1:
            W[i] += 1.0
        else:
            W[j] += 1.0

    # Initialize uniform strengths
    p = [1.0 / n] * n

    for _ in range(MAX_ITER):
        new_p = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j, nij in N[i].items():
                denom += nij / (p[i] + p[j])
            if denom > 0 and W[i] > 0:
                new_p[i] = W[i] / denom
            else:
                # Item with zero wins (or no comparisons) -> strength shrinks
                # toward zero; floor with a tiny epsilon to keep log finite.
                new_p[i] = 1e-12

        s = sum(new_p)
        new_p = [x / s for x in new_p]

        # L1 convergence on normalized strengths
        delta = sum(abs(new_p[i] - p[i]) for i in range(n))
        p = new_p
        if delta < TOL:
            break

    beta = [math.log(x) if x > 0 else float("-inf") for x in p]
    return p, beta



# NDCG

def dcg(rels: List[float]) -> float:
    """Standard DCG with log2(rank+1) discount, ranks are 1-indexed."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(relevances_in_pred_order: List[float], k: int) -> float:
    """NDCG@k for binary (or graded) relevance.

    `relevances_in_pred_order[t]` is the relevance of the item at predicted
    rank t+1. The ideal ordering is the same list sorted descending.
    """
    if k <= 0:
        return 0.0
    pred = relevances_in_pred_order[:k]
    ideal = sorted(relevances_in_pred_order, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(pred) / idcg



# Main

def main() -> None:
    combat_by_style = load_combat(COMBAT_FILE)
    judged = load_judged(JUDGED_FILE, n=100)

    # Order of items = order of question_i first appearance in combat file
    items: List[str] = []
    seen = set()
    with open(COMBAT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question_i"]
            if q not in seen:
                items.append(q)
                seen.add(q)

    # Sanity check: the first 100 rows of the judged file are exactly the
    # 100 question_i values from the combat file, in the same order.
    judged_qs = [r["question"] for r in judged]
    assert judged_qs == items, (
        "Mismatch: combat question_i order != first 100 of judged file. "
        "Check that data/qwen2.5results.csv_judged_results.csv is the file "
        "the combat run was generated from."
    )

    judged_by_q = {r["question"]: r for r in judged}

    # Output rows for the BTL score CSV
    rank_rows: List[Dict[str, str]] = []
    # NDCG@N per style
    ndcg_rows: List[Dict[str, str]] = []

    for style in STYLES:
        pairs = combat_by_style.get(style, [])
        p, beta = fit_btl(items, pairs)

        # Sort items by descending strength -> rank 1 = highest BTL
        order = sorted(range(len(items)), key=lambda k: p[k], reverse=True)

        # Per-style relevances in predicted order, for NDCG
        rels_in_order: List[float] = []
        for rank_idx, k in enumerate(order, start=1):
            q = items[k]
            row = judged_by_q[q]
            answer = row[f"qwen2.5 {style}"]
            verdict = row[f"judge_{style}"].strip().lower()
            result = 1 if verdict == "yes" else 0
            rels_in_order.append(float(result))
            rank_rows.append({
                "question": q,
                "style": style,
                "btl_rank": str(rank_idx),
                "btl_strength": f"{p[k]:.6f}",
                "btl_beta": f"{beta[k]:.6f}",
                "answer": answer,
                "result": str(result),
            })

        n = len(items)
        ndcg_full = ndcg_at_k(rels_in_order, n)
        ndcg_rows.append({
            "style": style,
            "n_items": str(n),
            "n_pairs": str(len(pairs)),
            "ndcg": f"{ndcg_full:.6f}",
            "ndcg_at_10": f"{ndcg_at_k(rels_in_order, 10):.6f}",
            "ndcg_at_25": f"{ndcg_at_k(rels_in_order, 25):.6f}",
            "ndcg_at_50": f"{ndcg_at_k(rels_in_order, 50):.6f}",
            "accuracy_top_10": f"{sum(rels_in_order[:10]) / 10:.3f}",
            "accuracy_top_25": f"{sum(rels_in_order[:25]) / 25:.3f}",
            "accuracy_overall": f"{sum(rels_in_order) / n:.3f}",
        })

    # Write outputs
    with open(OUT_RANK, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "question", "style", "btl_rank", "btl_strength",
            "btl_beta", "answer", "result",
        ])
        w.writeheader()
        w.writerows(rank_rows)

    with open(OUT_NDCG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "style", "n_items", "n_pairs", "ndcg",
            "ndcg_at_10", "ndcg_at_25", "ndcg_at_50",
            "accuracy_top_10", "accuracy_top_25", "accuracy_overall",
        ])
        w.writeheader()
        w.writerows(ndcg_rows)

    print(f"Wrote {len(rank_rows)} rows -> {OUT_RANK}")
    print(f"Wrote {len(ndcg_rows)} rows -> {OUT_NDCG}\n")
    print("NDCG by style:")
    print(f"{'style':<14} {'NDCG':>8} {'@10':>8} {'@25':>8} {'@50':>8} "
          f"{'acc@10':>8} {'acc@25':>8} {'acc':>6}")
    for r in ndcg_rows:
        print(f"{r['style']:<14} {r['ndcg']:>8} {r['ndcg_at_10']:>8} "
              f"{r['ndcg_at_25']:>8} {r['ndcg_at_50']:>8} "
              f"{r['accuracy_top_10']:>8} {r['accuracy_top_25']:>8} "
              f"{r['accuracy_overall']:>6}")


if __name__ == "__main__":
    main()