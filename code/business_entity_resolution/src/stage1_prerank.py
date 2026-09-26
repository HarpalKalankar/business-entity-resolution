"""Stage 1b: cheap pre-ranker -> final candidate set (candidate_pairs.tsv).

The blocking union has ~50 pairs per S1 entity. A small LightGBM on cheap features
(blocking score/ranks + a few fast similarities) keeps the top-K per S1 entity across
S2 and S3. That pruned list is exactly what Stage 2 scores, so it is what
candidate_pairs.tsv contains.

Train (on train candidates, holds out 20%% of S1 entities to measure recall@K):
    python src/stage1_prerank.py --work ../../work --mode train --suffix _s2
Apply (test) -> work/cands/test_final{suffix}.parquet and output/candidate_pairs{suffix}.tsv:
    python src/stage1_prerank.py --work ../../work --mode apply --split test --suffix _s2 --k 25
"""
import argparse
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import load_frame, save_parquet, shard_tag  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import in_train_frac  # noqa: E402

SIDE_COLS = ["entity_id", "name_core", "name_key", "legal", "house_num", "all_nums", "street", "city",
             "state", "addr_clean", "addr_missing"]
FEATURES = ["score", "fwd_rank", "rev_rank", "src", "name_key_eq", "name_jacc", "name_tsr", "name_ratio",
            "legal_eq", "house_eq", "house_missing", "nums_overlap", "street_jacc", "city_eq", "state_eq",
            "addr_jacc", "addr_missing_x", "score_gap"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _jacc(a, b):
    if not a or not b:
        return -1.0
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b)


def cheap_features(c: pd.DataFrame, L: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """c: candidate pairs (s1_id, x_id, src, score, fwd_rank, rev_rank). L/R: normalized S1 / S2+S3."""
    li = L.index.get_indexer(c.s1_id)
    ri = R.index.get_indexer(c.x_id)
    g = lambda df, col, idx: df[col].values[idx]  # noqa: E731
    f = pd.DataFrame({"score": c.score.values, "fwd_rank": c.fwd_rank.values, "rev_rank": c.rev_rank.values,
                      "src": c.src.values})
    ak, bk = g(L, "name_key", li), g(R, "name_key", ri)
    ac, bc = g(L, "name_core", li), g(R, "name_core", ri)
    f["name_key_eq"] = (ak == bk).astype(np.int8)
    f["name_jacc"] = [_jacc(a, b) for a, b in zip(ac, bc)]
    f["name_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(ac, bc)]
    f["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(ak, bk)]
    al, bl = g(L, "legal", li), g(R, "legal", ri)
    f["legal_eq"] = np.where((al == "") | (bl == ""), -1, (al == bl).astype(np.int8))
    ah, bh = g(L, "house_num", li), g(R, "house_num", ri)
    f["house_missing"] = ((ah == "") | (bh == "")).astype(np.int8)
    f["house_eq"] = np.where(f.house_missing == 1, -1, (ah == bh).astype(np.int8))
    f["nums_overlap"] = [_jacc(a, b) for a, b in zip(g(L, "all_nums", li), g(R, "all_nums", ri))]
    f["street_jacc"] = [_jacc(a, b) for a, b in zip(g(L, "street", li), g(R, "street", ri))]
    ac_, bc_ = g(L, "city", li), g(R, "city", ri)
    f["city_eq"] = np.where((ac_ == "") | (bc_ == ""), -1, (ac_ == bc_).astype(np.int8))
    as_, bs_ = g(L, "state", li), g(R, "state", ri)
    f["state_eq"] = np.where((as_ == "") | (bs_ == ""), -1, (as_ == bs_).astype(np.int8))
    f["addr_jacc"] = [_jacc(a, b) for a, b in zip(g(L, "addr_clean", li), g(R, "addr_clean", ri))]
    f["addr_missing_x"] = g(R, "addr_missing", ri)
    best = c.groupby("s1_id").score.transform("max").values
    f["score_gap"] = best - c.score.values
    return f


def load_sides(nd, split, suffix, cands=None):
    """Normalized S1 / S2+S3 rows; only those referenced by `cands` when given (RAM)."""
    ids1 = None if cands is None else set(cands.s1_id.unique())
    idsx = None if cands is None else set(cands.x_id.unique())
    L = load_frame(os.path.join(nd, f"{split}_source1{suffix}.parquet"), SIDE_COLS, ids1).set_index("entity_id")
    R = pd.concat([load_frame(os.path.join(nd, f"{split}_source{k}{suffix}.parquet"), SIDE_COLS, idsx)
                   for k in (2, 3)]).set_index("entity_id")
    return L, R


def featurize(cands, L, R, chunk=2_000_000):
    parts = []
    for s in range(0, len(cands), chunk):
        parts.append(cheap_features(cands.iloc[s:s + chunk], L, R))
        log(f"  features {min(s + chunk, len(cands)):,}/{len(cands):,}")
    return pd.concat(parts, ignore_index=True)


def topk(cands, prob, k):
    c = cands.assign(p=prob)
    c["r"] = c.groupby("s1_id").p.rank(ascending=False, method="first")
    return c[c.r <= k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", choices=["train", "apply"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--k", type=int, default=25, help="candidates kept per S1 entity (S2+S3 together)")
    ap.add_argument("--out", default=None, help="output folder for candidate_pairs.tsv (default <work>/../output)")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    nd, cd = os.path.join(args.work, "norm"), os.path.join(args.work, "cands")
    md = os.path.join(args.work, "models")
    os.makedirs(md, exist_ok=True)
    model_path = os.path.join(md, "prerank.txt")

    split = "train" if args.mode == "train" else args.split
    tag = shard_tag(args.shard, args.n_shards) if args.mode == "apply" else ""
    cands = pd.read_parquet(os.path.join(cd, f"{split}_cands{args.suffix}{tag}.parquet"))
    L, R = load_sides(nd, split, args.suffix, cands)
    if split == "train" and args.train_frac < 1:
        L = L[[in_train_frac(e, args.train_frac) for e in L.index]]
    log(f"{split}: {len(cands):,} candidate pairs")
    X = featurize(cands, L, R)

    if args.mode == "train":
        gt = pd.read_csv(os.path.join(nd, f"train_ground_truth{args.suffix}.tsv") if args.suffix else
                         os.path.join(args.work, "..", "dataset", "train", "train_ground_truth.tsv"),
                         sep="\t", dtype=str, keep_default_na=False)
        gt = gt[gt.source1_entity_id.isin(L.index)]
        true = {(a, x) for a, ids in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in ids.split(",") if x}
        y = np.fromiter(((a, x) in true for a, x in zip(cands.s1_id, cands.x_id)), bool, len(cands)).astype(np.int8)
        hold = cands.s1_id.map(lambda e: zlib.crc32(e.encode()) % 5 == 0).values
        params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50,
                      feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
        dtr = lgb.Dataset(X[FEATURES][~hold], y[~hold])
        dva = lgb.Dataset(X[FEATURES][hold], y[hold])
        m = lgb.train(params, dtr, 300, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
        log(f"pre-ranker trained: {m.best_iteration} trees")
        # recall@K on held-out entities
        hc = cands[hold].reset_index(drop=True)
        p = m.predict(X[FEATURES][hold], num_iteration=m.best_iteration)
        ents = gt[gt.source1_entity_id.map(lambda e: zlib.crc32(e.encode()) % 5 == 0)]
        n_true = ents.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
        tot = int(n_true.sum())
        print(f"held-out S1 entities {len(ents):,}, true pairs {tot:,}, union recall "
              f"{sum((a, x) in true for a, x in zip(hc.s1_id, hc.x_id)) / tot:.2%} at {len(hc) / len(ents):.1f}/S1")
        for k in (5, 10, 15, 20, 25, 30, 40):
            t = topk(hc, p, k)
            hit = np.fromiter(((a, x) in true for a, x in zip(t.s1_id, t.x_id)), bool, len(t))
            found = pd.Series(hit, index=t.s1_id.values).groupby(level=0).sum().reindex(n_true.index, fill_value=0)
            r = np.where(n_true > 0, found / np.maximum(n_true, 1), 1.0)
            f05 = np.where(n_true > 0, np.where(r > 0, 1.25 * r / (0.25 + r), 0), 1).mean()
            print(f"  K={k:<3} pairs/S1 {len(t) / len(ents):5.1f} | pair recall {hit.sum() / tot:6.2%} | oracle F0.5 {f05:.4f}")
        imp = pd.Series(m.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
        print("feature gain:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items()))
        m.save_model(model_path, num_iteration=m.best_iteration)
        log(f"saved {model_path}")
    else:
        m = lgb.Booster(model_file=model_path)
        p = m.predict(X[FEATURES])
        final = topk(cands, p, args.k)[["s1_id", "x_id", "src", "score", "fwd_rank", "rev_rank", "p"]]
        final = final.rename(columns={"p": "prerank_p"})
        dst = os.path.join(cd, f"{split}_final{args.suffix}{tag}.parquet")
        save_parquet(final, dst)
        log(f"wrote {dst}: {len(final):,} pairs ({len(final) / len(L):.1f}/S1)")
        if split == "test" and args.n_shards <= 1:   # sharded runs: stage3_merge writes the TSV
            out_dir = args.out or os.path.join(args.work, "..", "output")
            os.makedirs(out_dir, exist_ok=True)
            lists = final.sort_values(["s1_id", "prerank_p"], ascending=[True, False]).groupby("s1_id").x_id.agg(",".join)
            all_s1 = load_frame(os.path.join(nd, f"{split}_source1{args.suffix}.parquet"), ["entity_id"]).entity_id
            rows = pd.DataFrame({"source1_entity_id": all_s1.values})
            rows["candidate_entity_ids"] = rows.source1_entity_id.map(lists).fillna("")
            tsv = os.path.join(out_dir, f"candidate_pairs{args.suffix}.tsv")
            rows.to_csv(tsv, sep="\t", index=False)
            log(f"wrote {tsv}: {len(rows):,} S1 rows ({(rows.candidate_entity_ids == '').sum():,} empty)")


if __name__ == "__main__":
    main()
