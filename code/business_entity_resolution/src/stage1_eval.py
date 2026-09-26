"""Evaluate Stage 1 blocking on the TRAIN split (needs ground truth).

Reports pair recall, entity-level coverage, candidates per S1 and the oracle F0.5 ceiling
(the macro F0.5 a perfect matcher would get using only these candidates), for several
candidate caps, so the K / k_rev trade-off can be chosen.

Usage:  python src/stage1_eval.py --work ../../work --suffix _s2
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

from sampling import in_train_frac


def oracle_f05(n_true, n_found):
    """Per-entity F0.5 when precision is perfect; singletons (n_true=0) score 1."""
    r = np.where(n_true > 0, n_found / np.maximum(n_true, 1), 1.0)
    f = np.where(n_true > 0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
    return f.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--train-frac", type=float, default=1.0)
    args = ap.parse_args()
    nd = os.path.join(args.work, "norm")
    cands = pd.read_parquet(os.path.join(args.work, "cands", f"train_cands{args.suffix}.parquet"))
    gt = pd.read_csv(os.path.join(nd, f"train_ground_truth{args.suffix}.tsv") if args.suffix
                     else os.path.join(args.work, "..", "dataset", "train", "train_ground_truth.tsv"),
                     sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_parquet(os.path.join(nd, f"train_source1{args.suffix}.parquet"), columns=["entity_id", "country"])
    if args.train_frac < 1:
        s1 = s1[[in_train_frac(x, args.train_frac) for x in s1.entity_id]]
    gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
    cands = cands[cands.s1_id.isin(s1.entity_id)]
    true = pd.DataFrame([(a, x) for a, ids in zip(gt.source1_entity_id, gt.matched_entity_ids)
                         for x in ids.split(",") if x], columns=["s1_id", "x_id"])
    true["src"] = true.x_id.str[1].astype(int)
    n_true = gt.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    country = s1.set_index("entity_id").country
    print(f"S1 entities: {len(gt):,}   true pairs: {len(true):,}   singletons: {(n_true == 0).mean():.1%}")

    def report(c, label):
        hit = true.merge(c[["s1_id", "x_id"]], on=["s1_id", "x_id"], how="inner")
        found = hit.groupby("s1_id").size().reindex(n_true.index, fill_value=0)
        full = ((found == n_true) | (n_true == 0)).mean()
        print(f"{label:<28} pairs/S1 {len(c) / len(gt):6.1f} | pair recall {len(hit) / len(true):6.2%} | "
              f"entities fully covered {full:6.2%} | oracle F0.5 {oracle_f05(n_true.values, found.values):.4f}")
        return hit

    print("\n--- candidate caps (per source: forward top-K  OR  reverse top-k) ---")
    for K in (5, 10, 20, 30):
        for kr in (0, 1, 3):
            sub = cands[(cands.fwd_rank <= K) | (cands.rev_rank <= kr)]
            report(sub, f"K={K:<3} k_rev={kr}")
    hit = report(cands, "ALL")

    print("\n--- recall by country / source (ALL) ---")
    t = true.assign(c=true.s1_id.map(country)).merge(hit.assign(h=1), on=["s1_id", "x_id", "src"], how="left")
    print(t.groupby(["c", "src"]).h.apply(lambda s: f"{s.notna().mean():.2%}").unstack().to_string())

    miss = t[t.h.isna()].sample(min(15, int(t.h.isna().sum())), random_state=0)
    if len(miss):
        a = pd.read_parquet(os.path.join(nd, f"train_source1{args.suffix}.parquet"),
                            columns=["entity_id", "name_raw", "addr_raw"]).set_index("entity_id")
        o = pd.concat([pd.read_parquet(os.path.join(nd, f"train_source{k}{args.suffix}.parquet"),
                                       columns=["entity_id", "name_raw", "addr_raw"]) for k in (2, 3)]).set_index("entity_id")
        print("\n--- sample of MISSED true pairs ---")
        for s, x in zip(miss.s1_id, miss.x_id):
            print(f"  S1: {a.name_raw[s]} | {a.addr_raw[s]}\n   X: {o.name_raw[x]} | {o.addr_raw[x]}")


if __name__ == "__main__":
    main()
