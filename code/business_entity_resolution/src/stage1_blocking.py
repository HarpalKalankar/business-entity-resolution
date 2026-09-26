"""Stage 1: candidate generation (blocking).

For every country label present in Source 1 (open set; records whose label is unseen are
compared against all of Source 1), and separately for Source 2 and Source 3:

  1. build hashed feature bags (block_features.py)
  2. drop features whose frequency in the target source exceeds a cap (bounds fan-out)
  3. score = sum of IDF of shared features, via sparse matrix product (sparse_dot_topn)
  4. forward:  top-K targets per S1 entity
     reverse:  top-k S1 entities per target record  (each S2/S3 record has <= 1 owner,
               so its own best S1 is a strong candidate even if that S1 has many look-alikes)
  5. union -> work/cands/{split}_cands{suffix}.parquet

Usage (from code/business_entity_resolution):
    python src/stage1_blocking.py --work ../../work --split train --suffix _s2
    python src/stage1_blocking.py --work ../../work --split test
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import in_train_frac, save_parquet  # noqa: E402
from block_features import FAM_W, frame_features  # noqa: E402
from sampling import in_shard, load_frame, shard_tag  # noqa: E402

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None

COLS = ["entity_id", "country", "name_core", "name_key", "name_alt", "street", "city", "state", "house_num",
        "name_clean", "locality", "unit", "addr_clean"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def topn(A, B, n, threads):
    """Row-wise top-n of A @ B (both CSR). Returns COO arrays (row, col, score)."""
    if sp_matmul_topn is not None:
        C = sp_matmul_topn(A, B, top_n=n, threshold=0.0, sort=True, n_threads=threads)
    else:  # pure-scipy fallback (slower), chunked
        r, c, v = [], [], []
        for s0 in range(0, A.shape[0], 2000):
            C = (A[s0:s0 + 2000] @ B).tocsr()
            for i in range(C.shape[0]):
                lo, hi = C.indptr[i], C.indptr[i + 1]
                if hi > lo:
                    o = np.argsort(-C.data[lo:hi])[:n]
                    r.append(np.full(len(o), i + s0)); c.append(C.indices[lo:hi][o]); v.append(C.data[lo:hi][o])
        if not r:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return (np.concatenate(r).astype(np.int64), np.concatenate(c).astype(np.int64),
                np.concatenate(v).astype(np.float32))
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def _jitter(ids) -> np.ndarray:
    """1 + ~1e-6 * hash(id): deterministic tie-breaker, so top-k cut-offs are identical
    whether the S1 entities are processed in one go or in shards."""
    h = pd.util.hash_array(np.asarray(ids, dtype=object)).astype(np.float64) / 2.0**64
    return (1.0 + 1e-6 * h).astype(np.float32)


def block(s1: pd.DataFrame, nx: int, xfeat, k_fwd: int, k_rev: int, df_cap: int, threads: int, x_ids=None):
    """Candidate pairs between one S1 frame and one target group of nx records (same country).
    xfeat = (row, hash) arrays of the target records, precomputed once and cached."""
    if len(s1) == 0 or nx == 0:
        return pd.DataFrame(columns=["i1", "ix", "score", "fwd_rank", "rev_rank"])
    r1, h1, f1 = frame_features(s1)
    r2, h2 = xfeat
    # unique (row, feature) on target side, then document frequency per feature
    key2 = np.unique(np.stack([h2, r2.astype(np.uint64)], 1), axis=0)
    h2u, r2u = key2[:, 0], key2[:, 1].astype(np.int64)
    feats, inv, dfs = np.unique(h2u, return_inverse=True, return_counts=True)
    keep = dfs <= df_cap
    idf = np.log1p(nx / dfs).astype(np.float32)
    idf[~keep] = 0.0
    B = sp.csr_matrix((np.ones(len(r2u), np.float32) * keep[inv], (r2u, inv)), shape=(nx, len(feats)))
    B.eliminate_zeros()
    if x_ids is not None:
        B = sp.diags(_jitter(x_ids)) @ B
    # S1 side: map to target vocabulary, weight = family multiplier x IDF, max over duplicates
    pos = np.searchsorted(feats, h1)
    pos[pos >= len(feats)] = 0
    ok = (feats[pos] == h1) & keep[pos]
    w = FAM_W[f1[ok]] * idf[pos[ok]]
    A = sp.coo_matrix((w, (r1[ok], pos[ok])), shape=(len(s1), len(feats))).tocsr()
    A.sum_duplicates()  # a feature repeated in one record counts once more; acceptable
    A = (sp.diags(_jitter(s1.entity_id.values)) @ A).tocsr()
    B = B.tocsr()
    Bt = B.T.tocsr()
    fr, fc, fv = topn(A, Bt, k_fwd, threads)                 # S1 -> targets
    rr, rc, rv = topn(B, A.T.tocsr(), k_rev, threads)        # target -> S1
    fwd = pd.DataFrame({"i1": fr, "ix": fc, "score": fv})
    fwd["fwd_rank"] = fwd.groupby("i1")["score"].rank(ascending=False, method="first").astype(np.int16)
    rev = pd.DataFrame({"i1": rc, "ix": rr, "score": rv})
    rev["rev_rank"] = rev.groupby("ix")["score"].rank(ascending=False, method="first").astype(np.int16)
    fwd["rev_rank"] = np.int16(999)
    rev["fwd_rank"] = np.int16(999)
    out = (pd.concat([fwd, rev], ignore_index=True)
             .groupby(["i1", "ix"], as_index=False)
             .agg(score=("score", "max"), fwd_rank=("fwd_rank", "min"), rev_rank=("rev_rank", "min")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--suffix", default="", help="e.g. _s2 for the 2%% dev sample")
    ap.add_argument("--k-fwd", type=int, default=30, help="top-K targets per S1 entity, per source")
    ap.add_argument("--k-rev", type=int, default=3, help="top-k S1 entities per target record")
    ap.add_argument("--df-cap", type=int, default=2000,
                    help="drop features seen in more target records than this (full-data scale)")
    ap.add_argument("--sample-pct", type=float, default=None,
                    help="sample %% used in stage 0 (scales --df-cap); inferred from suffix _sN")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1, help="process S1 in N hash shards (RAM)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="train split only: use this share of S1 entities (still blocked against FULL S2/S3 pools)")
    args = ap.parse_args()

    pct = args.sample_pct
    if pct is None:
        pct = float(args.suffix[2:]) if args.suffix.startswith("_s") else 100.0
    cap = max(5, int(round(args.df_cap * pct / 100)))
    nd = os.path.join(args.work, "norm")
    out_dir = os.path.join(args.work, "cands")
    os.makedirs(out_dir, exist_ok=True)

    s1 = load_frame(os.path.join(nd, f"{args.split}_source1{args.suffix}.parquet"), COLS)
    if args.split == "train" and args.train_frac < 1:
        s1 = s1[[in_train_frac(e, args.train_frac) for e in s1.entity_id]].reset_index(drop=True)
    if args.n_shards > 1:
        s1 = s1[[in_shard(e, args.shard, args.n_shards) for e in s1.entity_id]].reset_index(drop=True)
    tag = shard_tag(args.shard, args.n_shards)
    log(f"S1 {len(s1):,} rows{' (shard ' + str(args.shard) + '/' + str(args.n_shards) + ')' if tag else ''}; "
        f"df cap {cap} (pct {pct:g})")
    s1_countries = sorted(set(s1["country"]))
    results = []

    def emit(a, ids_b, pr, k):
        return pd.DataFrame({"s1_id": a.entity_id.values[pr.i1.values.astype(np.int64)],
                             "x_id": ids_b[pr.ix.values.astype(np.int64)], "src": np.int8(k),
                             "score": pr.score.values.astype(np.float32),
                             "fwd_rank": pr.fwd_rank.values.astype(np.int16),
                             "rev_rank": pr.rev_rank.values.astype(np.int16)})

    for k in (2, 3):
        xpath = os.path.join(nd, f"{args.split}_source{k}{args.suffix}.parquet")
        xc = load_frame(xpath, ["entity_id", "country"])
        groups = [(c, xc.country.values == c) for c in s1_countries]
        other = ~xc.country.isin(s1_countries).values
        if other.any():
            groups.append(("__other__", other))  # open set: unseen labels -> against all S1
        for c, mask in groups:
            ids_b = xc.entity_id.values[mask]
            a = s1 if c == "__other__" else s1[s1.country == c].reset_index(drop=True)
            if len(ids_b) == 0 or len(a) == 0:
                continue
            t0 = time.time()
            xfeat = cached_target_features(out_dir, xpath, args.split, args.suffix, k, c, ids_b)
            pr = block(a, len(ids_b), xfeat, args.k_fwd, args.k_rev, cap, args.threads, x_ids=ids_b)
            results.append(emit(a, ids_b, pr, k))
            log(f"  source{k} {c:>9}: S1 {len(a):,} x {len(ids_b):,} -> {len(pr):,} pairs "
                f"({len(pr) / max(1, len(a)):.1f}/S1)  {time.time() - t0:.0f}s")
        del xc
    cands = pd.concat(results, ignore_index=True)
    dst = os.path.join(out_dir, f"{args.split}_cands{args.suffix}{tag}.parquet")
    save_parquet(cands, dst)
    log(f"wrote {dst}: {len(cands):,} pairs, {cands.s1_id.nunique():,} S1 with >=1 candidate")


def cached_target_features(out_dir, xpath, split, suffix, k, country, ids_b):
    """Blocking features of one (source, country) group, computed once and reused by every shard."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    cache = os.path.join(out_dir, "xfeat", f"{split}{suffix}_s{k}_{safe}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if int(z["n"]) == len(ids_b):
            return z["r"], z["h"]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    x = load_frame(xpath, COLS, country=None if country == "__other__" else country)
    if country == "__other__":
        x = x[x.entity_id.isin(set(ids_b))]
    x = x.set_index("entity_id").loc[ids_b].reset_index()   # same row order as ids_b
    r, h, _ = frame_features(x)
    del x
    np.savez(cache + ".tmp.npz", r=r, h=h, n=np.int64(len(ids_b)))
    os.replace(cache + ".tmp.npz", cache)
    return r, h


if __name__ == "__main__":
    main()
