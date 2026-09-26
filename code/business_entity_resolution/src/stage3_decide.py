"""Stage 3: decision layer -> matching_results.tsv

Turns pair probabilities into one match list per Source 1 entity, precision first:

  1. calibration   isotonic regression fitted on out-of-fold train probabilities
  2. one owner     every S2/S3 record goes only to its best S1 entity (never 2 entities),
                   optionally only if it beats the runner-up by `margin`
  3. expected F0.5 per S1 entity, candidates sorted p1 >= p2 >= ...; predicting the top k gives
                     TP = sum(p_1..p_k), FP = k - TP, FN = sum(p_k+1..p_n)
                     E[F0.5](k) = 1.25 TP / (1.25 TP + 0.25 FN + FP)
                   predicting nothing scores 1 only for a singleton:  E[F0.5](0) = prod(1 - p_i)
                   pick the best k (k = 0 means an empty list)
     knobs         p_min (never add a pair below it) and alpha (weight on the empty option),
                   tuned on held-out train entities for macro F0.5

Tune (train OOF from stage 2):
    python src/stage3_decide.py --work ../../work --suffix _s2 --mode tune
Apply (test scores from stage 2) -> output/matching_results{suffix}.tsv:
    python src/stage3_decide.py --work ../../work --suffix _s2 --mode apply
"""
import argparse
import itertools
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import in_train_frac, save_parquet  # noqa: E402
from stage2_matcher import load_gt, macro_f05  # noqa: E402
from sampling import shard_tag  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------ decisions ---
def one_owner(d: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Keep each x only for its best S1, and only if it beats the runner-up S1 by `margin`."""
    d = d.sort_values(["x_id", "p"], ascending=[True, False])
    r = d.groupby("x_id").cumcount()
    second = d.x_id.map(d[r == 1].set_index("x_id").p).fillna(0.0)
    keep = (r == 0) & (d.p - second >= margin)
    return d[keep.values]


def expected_f05_select(d: pd.DataFrame, p_min: float, alpha: float) -> pd.DataFrame:
    """Per S1 entity choose the prefix (by p) that maximises expected F0.5, or nothing."""
    if d.empty:
        return d
    d = d.sort_values(["s1_id", "p"], ascending=[True, False]).copy()
    g = d.groupby("s1_id", sort=False)
    d["k"] = g.cumcount() + 1
    d["tp"] = g.p.cumsum()
    d["tot"] = g.p.transform("sum")
    fp, fn = d.k - d.tp, d.tot - d.tp
    d["ef"] = np.where(d.p >= p_min, 1.25 * d.tp / (1.25 * d.tp + 0.25 * fn + fp), -1.0)
    logq = np.log(np.clip(1 - d.p.values, 1e-9, 1))
    d["e0"] = alpha * np.exp(pd.Series(logq, index=d.index).groupby(d.s1_id).transform("sum"))
    best = d.loc[d.groupby("s1_id", sort=False).ef.idxmax(), ["s1_id", "k", "ef", "e0"]]
    best = best[best.ef > best.e0]                     # otherwise: empty list (singleton)
    kmap = best.set_index("s1_id").k
    d["kbest"] = d.s1_id.map(kmap)
    return d[d.k <= d.kbest.fillna(0)][["s1_id", "x_id", "p"]]


def decide(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = one_owner(d, cfg["margin"])
    if cfg.get("rule") == "threshold":
        return d[d.p >= cfg["t"]][["s1_id", "x_id", "p"]]
    return expected_f05_select(d, cfg["p_min"], cfg["alpha"])


class Calibrator:
    def __init__(self, x=None, y=None):
        self.x, self.y = x, y

    def fit(self, p, y):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        self.x, self.y = ir.X_thresholds_.tolist(), ir.y_thresholds_.tolist()
        return self

    def __call__(self, p):
        return np.interp(p, self.x, self.y)


# ----------------------------------------------------------------- tune ---
GRID = dict(p_min=[0.3, 0.4, 0.5, 0.6, 0.7], alpha=[0.8, 1.0, 1.2, 1.5, 2.0], margin=[0.0, 0.1, 0.2],
            t=[0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])


class FastEval:
    """Precomputed arrays so every (p_min, alpha) setting is scored with a few numpy reductions.
    Gives exactly the same decisions as decide() + macro_f05()."""

    def __init__(self, d: pd.DataFrame, gt_n: pd.Series):
        d = d.sort_values(["s1_id", "p"], ascending=[True, False]).reset_index(drop=True)
        g = d.groupby("s1_id", sort=False)
        k = (g.cumcount() + 1).values
        tp = g.p.cumsum().values
        tot = g.p.transform("sum").values
        self.p, self.y, self.k = d.p.values, d.y.values.astype(np.int32), k
        self.ef = 1.25 * tp / (1.25 * tp + 0.25 * (tot - tp) + (k - tp))
        self.starts = np.flatnonzero(np.r_[True, d.s1_id.values[1:] != d.s1_id.values[:-1]])
        self.gid = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(d)]))
        self.e0 = np.exp(np.add.reduceat(np.log(np.clip(1 - self.p, 1e-9, 1)), self.starts))
        ent = d.s1_id.values[self.starts]
        self.nt_g = gt_n.reindex(ent).fillna(0).values
        rest = gt_n[~gt_n.index.isin(ent)]
        self.rest_score = float((rest == 0).sum())          # no candidates -> empty list
        self.n_all = len(gt_n)
        self.single = gt_n.values == 0

    def _f(self, sel):
        tp = np.add.reduceat(self.y * sel, self.starts)
        npred = np.add.reduceat(sel.astype(np.int32), self.starts)
        nt = self.nt_g
        P = np.divide(tp, npred, out=np.zeros(len(nt)), where=npred > 0)
        R = np.divide(tp, nt, out=np.zeros(len(nt)), where=nt > 0)
        f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(len(nt)), where=(0.25 * P + R) > 0)
        f = np.where(nt == 0, (npred == 0).astype(float), f)
        return (f.sum() + self.rest_score) / self.n_all

    def score_threshold(self, t):
        return self._f(self.p >= t)

    def score(self, p_min, alpha):
        efm = np.where(self.p >= p_min, self.ef, -1.0)
        gbest = np.maximum.reduceat(efm, self.starts)
        kbest = np.minimum.reduceat(np.where(efm == gbest[self.gid], self.k, 10**6), self.starts)
        use = gbest > alpha * self.e0
        sel = (self.k <= kbest[self.gid]) & use[self.gid]
        tp = np.add.reduceat(self.y * sel, self.starts)
        npred = np.add.reduceat(sel.astype(np.int32), self.starts)
        nt = self.nt_g
        P = np.divide(tp, npred, out=np.zeros(len(nt)), where=npred > 0)
        R = np.divide(tp, nt, out=np.zeros(len(nt)), where=nt > 0)
        f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(len(nt)), where=(0.25 * P + R) > 0)
        f = np.where(nt == 0, (npred == 0).astype(float), f)
        return (f.sum() + self.rest_score) / self.n_all


def step_tune(args):
    md = os.path.join(args.work, "models")
    oof = pd.read_parquet(os.path.join(args.work, "stage2", f"train_oof{args.suffix}.parquet"))
    s1 = pd.read_parquet(os.path.join(args.work, "norm", f"train_source1{args.suffix}.parquet"), columns=["entity_id"]).entity_id
    if args.train_frac < 1:
        s1 = s1[[in_train_frac(e, args.train_frac) for e in s1]]
    gt_n, gt_pairs = load_gt(args, s1)
    half = np.array([zlib.crc32(("h" + e).encode()) % 2 for e in gt_n.index])
    nA, nB = gt_n[half == 0], gt_n[half == 1]
    oA, oB = oof[oof.s1_id.isin(nA.index)], oof[oof.s1_id.isin(nB.index)]
    log(f"tuning on {len(nA):,} entities, reporting on {len(nB):,} untouched entities")

    results = []
    for col in ("p1", "p2"):
        cal = Calibrator().fit(oA[col].values, oA.y.values)
        best = None
        for mg in GRID["margin"]:
            dA = one_owner(oA[["s1_id", "x_id", "y"]].assign(p=cal(oA[col].values)), mg)
            fe = FastEval(dA, nA)
            for pm, al in itertools.product(GRID["p_min"], GRID["alpha"]):
                f = fe.score(pm, al)
                if best is None or f > best[0]:
                    best = (f, {"rule": "expected_f05", "p_min": pm, "alpha": al, "margin": mg})
            for t in GRID["t"]:
                f = fe.score_threshold(t)
                if f > best[0]:
                    best = (f, {"rule": "threshold", "t": t, "margin": mg})
        cfg = best[1]
        dB = oB[["s1_id", "x_id", "y"]].assign(p=cal(oB[col].values))
        fB = macro_f05(nB, gt_pairs, decide(dB, cfg))
        thr = one_owner(oB[["s1_id", "x_id"]].assign(p=oB[col].values), 0.0)
        fT = macro_f05(nB, gt_pairs, thr[thr.p >= 0.7])
        print(f"{col}: best on tune half {best[0]:.5f} with {cfg}")
        print(f"     held-out half: expected-F0.5 rule {fB[0]:.5f} (singletons {fB[1]:.4f}, matched {fB[2]:.5f})"
              f"  |  baseline one-owner + threshold 0.7: {fT[0]:.5f} (singletons {fT[1]:.4f}, matched {fT[2]:.5f})")
        results.append((best[0], fB[0], col, cfg))
    _, fB, col, cfg = max(results, key=lambda r: r[0])   # choose on the tuning half only
    cal = Calibrator().fit(oof[col].values, oof.y.values)   # final calibration on all entities
    out = {"prob": col, **cfg, "cal_x": cal.x, "cal_y": cal.y, "heldout_macro_f05": fB}
    with open(os.path.join(md, "stage3.json"), "w") as f:
        json.dump(out, f)
    print(f"\nchosen: {col} {cfg}  held-out macro F0.5 {fB:.5f}")
    log(f"saved {os.path.join(md, 'stage3.json')}")


# ---------------------------------------------------------------- apply ---
def step_apply(args):
    with open(os.path.join(args.work, "models", "stage3.json")) as f:
        cfg = json.load(f)
    cal = Calibrator(cfg["cal_x"], cfg["cal_y"])
    tag = shard_tag(args.shard, args.n_shards)
    s = pd.read_parquet(os.path.join(args.work, "stage2", f"test_scores{args.suffix}{tag}.parquet"))
    d = s[["s1_id", "x_id"]].assign(p=cal(s[cfg["prob"]].values))
    m = decide(d, cfg)
    if args.n_shards > 1:   # stage3_merge.py resolves records claimed by two shards and writes the TSVs
        os.makedirs(os.path.join(args.work, "stage3"), exist_ok=True)
        dst = os.path.join(args.work, "stage3", f"test_matches{args.suffix}{tag}.parquet")
        save_parquet(m, dst)
        log(f"wrote {dst}: {len(m):,} matches for {m.s1_id.nunique():,} S1")
        return
    s1 = pd.read_parquet(os.path.join(args.work, "norm", f"test_source1{args.suffix}.parquet"), columns=["entity_id", "country"])
    lists = m.sort_values(["s1_id", "p"], ascending=[True, False]).groupby("s1_id").x_id.agg(",".join)
    rows = pd.DataFrame({"source1_entity_id": s1.entity_id.values})
    rows["matched_entity_ids"] = rows.source1_entity_id.map(lists).fillna("")
    # --- self-checks (the official validator runs on the full test set)
    assert rows.source1_entity_id.is_unique
    allx = set(s.x_id)
    for ids in rows.matched_entity_ids:
        if ids:
            lst = ids.split(",")
            assert len(lst) == len(set(lst)) and all(x[:3] in ("S2-", "S3-") and x in allx for x in lst)
    out_dir = args.out or os.path.join(args.work, "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"matching_results{args.suffix}.tsv")
    rows.to_csv(dst, sep="\t", index=False)
    n = rows.matched_entity_ids.str.len().gt(0)
    by_c = pd.Series(n.values, index=s1.country.values).groupby(level=0).mean()
    log(f"wrote {dst}: {len(rows):,} S1 rows, {n.mean():.1%} with >=1 match, "
        f"{m.shape[0] / max(1, n.sum()):.2f} matches per matched entity; share matched by country "
        + ", ".join(f"{k} {v:.1%}" for k, v in by_c.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--mode", choices=["tune", "apply"], required=True)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    {"tune": step_tune, "apply": step_apply}[args.mode](args)


if __name__ == "__main__":
    main()
