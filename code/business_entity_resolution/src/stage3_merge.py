"""Merge sharded test results into the two submission files.

  output/candidate_pairs.tsv   union of every shard's final candidate list (what Stage 2 scored)
  output/matching_results.tsv  every shard's matches; a record claimed by two shards keeps
                               only its highest-probability S1 (global one-owner rule)
Every test S1 entity gets exactly one row (empty list when nothing matched).

    python src/stage3_merge.py --work ../../work --n-shards 8 [--suffix _s2] [--out ../../output]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import load_frame, shard_tag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.work, "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    tags = [shard_tag(i, args.n_shards) for i in range(args.n_shards)]
    s1 = load_frame(os.path.join(args.work, "norm", f"test_source1{args.suffix}.parquet"), ["entity_id"]).entity_id

    fin = pd.concat([pd.read_parquet(os.path.join(args.work, "cands", f"test_final{args.suffix}{t}.parquet"),
                                     columns=["s1_id", "x_id", "prerank_p"]) for t in tags], ignore_index=True)
    mat = pd.concat([pd.read_parquet(os.path.join(args.work, "stage3", f"test_matches{args.suffix}{t}.parquet"))
                     for t in tags], ignore_index=True)
    before = len(mat)
    mat = mat.sort_values("p", ascending=False).drop_duplicates("x_id", keep="first")
    print(f"matches {before:,} -> {len(mat):,} after global one-owner ({before - len(mat):,} cross-shard conflicts)")

    def write(pairs, sort_col, col_name, fname):
        lists = pairs.sort_values(["s1_id", sort_col], ascending=[True, False]).groupby("s1_id").x_id.agg(",".join)
        rows = pd.DataFrame({"source1_entity_id": s1.values})
        rows[col_name] = rows.source1_entity_id.map(lists).fillna("")
        assert rows.source1_entity_id.is_unique
        path = os.path.join(out_dir, f"{fname}{args.suffix}.tsv")
        rows.to_csv(path, sep="\t", index=False)
        print(f"wrote {path}: {len(rows):,} rows, {(rows[col_name] != '').sum():,} non-empty")

    fin = fin.drop_duplicates(["s1_id", "x_id"])
    missing = set(zip(mat.s1_id, mat.x_id)) - set(zip(fin.s1_id, fin.x_id))
    assert not missing, f"{len(missing)} matches are not in the candidate set (pipeline bug)"
    write(fin, "prerank_p", "candidate_entity_ids", "candidate_pairs")
    write(mat, "p", "matched_entity_ids", "matching_results")


if __name__ == "__main__":
    main()
