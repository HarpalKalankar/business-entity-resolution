"""Sanity report for Stage 0 output.

  * field coverage per split/source/country (how often house_num, city, state ... are filled)
  * on TRAIN ground-truth pairs: how often normalized fields agree between S1 and its matches,
    compared with the raw strings. Higher agreement on true pairs = better normalization.

Usage:  python src/stage0_report.py --work ../../work [--suffix _s2]
"""
import argparse
import os

import pandas as pd

FIELDS = ["name_core", "name_key", "legal", "house_num", "street", "city", "state", "postcode", "unit"]


def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    nd = os.path.join(args.work, "norm")
    frames = {}
    for split in ("train", "test"):
        for k in (1, 2, 3):
            p = os.path.join(nd, f"{split}_source{k}{args.suffix}.parquet")
            if os.path.exists(p):
                frames[(split, k)] = pd.read_parquet(p)

    print("\n=== field coverage (% non-empty) ===")
    rows = []
    for (split, k), df in frames.items():
        for c, g in df.groupby("country"):
            r = {"split": split, "src": k, "country": c, "n": len(g)}
            for f in ["house_num", "street", "city", "state", "postcode", "unit", "landmark", "legal"]:
                r[f] = round(100 * (g[f] != "").mean(), 1)
            r["addr_missing"] = round(100 * g["addr_missing"].mean(), 1)
            r["name_native"] = round(100 * (g["name_translit"] > 0).mean(), 1)
            r["native_fallback"] = round(100 * (g["name_translit"] == 1).mean(), 2)
            r["domain"] = round(100 * g["name_is_domain"].mean(), 2)
            rows.append(r)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(rows).to_string(index=False))

    gt_path = os.path.join(nd, f"train_ground_truth{args.suffix}.tsv")
    if not os.path.exists(gt_path) and not args.suffix:
        gt_path = None
    if gt_path and ("train", 1) in frames:
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        pairs = [(a, x) for a, ids in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in ids.split(",") if x]
        pairs = pd.DataFrame(pairs, columns=["s1", "x"])
        s1 = frames[("train", 1)].set_index("entity_id")
        other = pd.concat([frames[k] for k in (("train", 2), ("train", 3)) if k in frames]).set_index("entity_id")
        pairs = pairs[pairs.s1.isin(s1.index) & pairs.x.isin(other.index)]
        A, B = s1.loc[pairs.s1].reset_index(), other.loc[pairs.x].reset_index()
        print(f"\n=== agreement on {len(pairs):,} TRUE train pairs (both sides non-empty) ===")
        res = {"raw name exact": (A.name_raw.str.lower() == B.name_raw.str.lower()).mean(),
               "raw addr exact": (A.addr_raw.str.lower() == B.addr_raw.str.lower()).mean()}
        for f in FIELDS:
            m = (A[f] != "") & (B[f] != "")
            res[f"{f} equal"] = ((A[f] == B[f]) & m).sum() / max(1, m.sum())
        res["name_core token-jaccard >= 0.5"] = (pd.Series([jacc(a, b) for a, b in zip(A.name_core, B.name_core)]) >= 0.5).mean()
        res["addr_clean token-jaccard >= 0.5"] = (pd.Series([jacc(a, b) for a, b in zip(A.addr_clean, B.addr_clean)]) >= 0.5).mean()
        for k_, v in res.items():
            print(f"  {k_:<34} {100 * v:5.1f}%")
        for country in sorted(A.country.unique()):
            m = (A.country == country).values
            print(f"  [{country}] name_key equal {100 * (A.name_key[m].values == B.name_key[m].values).mean():.1f}%  "
                  f"house_num equal {100 * (A.house_num[m].values == B.house_num[m].values).mean():.1f}%  "
                  f"city equal {100 * (A.city[m].values == B.city[m].values).mean():.1f}%")
        print("\n=== examples where name_key differs on a true pair ===")
        d = (A.name_key != B.name_key).values
        ex = pd.DataFrame({"s1": A.name_raw[d].values, "x": B.name_raw[d].values,
                           "s1_key": A.name_key[d].values, "x_key": B.name_key[d].values}).sample(20, random_state=0)
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            print(ex.to_string(index=False))


if __name__ == "__main__":
    main()
