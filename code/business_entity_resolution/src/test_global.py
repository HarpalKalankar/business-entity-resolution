"""Cross-shard corrections for the sharded test run (keeps sharded == unsharded).

Sharding S1 entities is exact for everything seen *from an S1 entity* (all its candidates
live in its shard), but a S2/S3 record's competitors are spread over all shards. These
passes recompute the record-side quantities over ALL shards using compact numeric arrays:

  revrank  after blocking:   global reverse rank of each pair by blocking score; drops pairs
                             that were only a shard-local reverse top-k
  c0       after pre-ranker: record-side competition from the pre-ranker probability
  m        after pass 1:     record-side competition from the pass-1 probability

    python src/test_global.py --work ../../work --n-shards 8 --step revrank|c0|m [--suffix _s2]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import save_parquet, shard_tag  # noqa: E402

X_COLS = ["rank_x", "gap_x", "n_x", "margin_x"]


def id_code(ids: pd.Series) -> np.ndarray:
    """'S2-192345572' -> 2*10^12 + 192345572 (int64): cheap, unique, no Python strings kept."""
    s = ids.astype(str)
    return s.str[1].astype(np.int64).values * 10**12 + s.str[3:].astype(np.int64).values


def ordinal_rank(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """1, 2, 3 ... within each x by v descending (ties broken by position, like a top-n list)."""
    o = np.lexsort((-v, x))
    xs = x[o]
    start = np.r_[True, xs[1:] != xs[:-1]]
    first = np.flatnonzero(start)
    pos = np.arange(len(xs)) - first[np.cumsum(start) - 1]
    r = np.empty(len(o), np.int64)
    r[o] = pos + 1
    return r


def x_side(x: np.ndarray, p: np.ndarray) -> dict:
    """Record-side competition, identical to pair_features.competition's *_x columns."""
    o = np.lexsort((-p, x))
    xs, ps = x[o], p[o]
    start = np.r_[True, xs[1:] != xs[:-1]]
    gidx = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, len(xs)])
    pos = np.arange(len(xs)) - first[gidx]
    mx = ps[first][gidx]
    sec = np.where(size > 1, ps[np.minimum(first + 1, len(ps) - 1)], 0.0)[gidx]
    # rank with ties = 'min': 1 + number of strictly larger p in the group
    rank = pos + 1
    tie = np.r_[False, (ps[1:] == ps[:-1]) & ~start[1:]]
    if tie.any():  # propagate the first rank of each tie run
        run_start = np.flatnonzero(~tie)
        run_id = np.cumsum(~tie) - 1
        rank = rank[run_start][run_id]
    out = {"rank_x": rank.astype(np.float32), "gap_x": (mx - ps).astype(np.float32),
           "n_x": size[gidx].astype(np.float32),
           "margin_x": np.where(rank == 1, ps - sec, ps - mx).astype(np.float32)}
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return {k: v[inv] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--step", choices=["revrank", "c0", "m"], required=True)
    ap.add_argument("--k-fwd", type=int, default=30)
    ap.add_argument("--k-rev", type=int, default=3)
    args = ap.parse_args()
    tags = [shard_tag(i, args.n_shards) for i in range(args.n_shards)]
    cd, sd = os.path.join(args.work, "cands"), os.path.join(args.work, "stage2")

    if args.step == "revrank":
        paths = [os.path.join(cd, f"test_cands{args.suffix}{t}.parquet") for t in tags]
        cols = ["x_id", "score"]
        parts = [pd.read_parquet(p, columns=cols) for p in paths]
        lens = [len(d) for d in parts]
        x = np.concatenate([id_code(d.x_id) for d in parts])
        sc = np.concatenate([d.score.values for d in parts]).astype(np.float64)
        del parts
        rr = ordinal_rank(x, sc)
        rr = np.where(rr <= args.k_rev, rr, 999)   # same meaning as unsharded: 999 = not in reverse top-k
        off = 0
        kept = total = 0
        for p, n in zip(paths, lens):
            d = pd.read_parquet(p)
            d["rev_rank"] = np.minimum(rr[off:off + n], 999).astype(np.int16)
            off += n
            d = d[(d.fwd_rank <= args.k_fwd) | (d.rev_rank <= args.k_rev)]
            total += n
            kept += len(d)
            save_parquet(d, p)
        print(f"global reverse rank: kept {kept:,} of {total:,} pairs")
        return

    if args.step == "c0":
        paths = [os.path.join(cd, f"test_final{args.suffix}{t}.parquet") for t in tags]
        pcol, prefix, outname = "prerank_p", "c0_", "test_c0x"
    else:
        paths = [os.path.join(sd, f"test_scores{args.suffix}{t}.parquet") for t in tags]
        pcol, prefix, outname = "p1", "m_", "test_mx"
    parts = [pd.read_parquet(p, columns=["x_id", pcol]) for p in paths]
    lens = [len(d) for d in parts]
    x = np.concatenate([id_code(d.x_id) for d in parts])
    pv = np.concatenate([d[pcol].values for d in parts]).astype(np.float64)
    del parts
    xs = x_side(x, pv)
    off = 0
    for t, n in zip(tags, lens):
        out = pd.DataFrame({prefix + k: v[off:off + n] for k, v in xs.items()})
        save_parquet(out, os.path.join(sd, f"{outname}{args.suffix}{t}.parquet"))
        off += n
    print(f"global record-side '{prefix}' competition written for {len(tags)} shards ({len(x):,} pairs)")


if __name__ == "__main__":
    main()
