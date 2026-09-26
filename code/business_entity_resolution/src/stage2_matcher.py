"""Stage 2: pair matcher (LightGBM, two passes, grouped K-fold).

  pass 1: pair features + competition from the pre-ranker probability
  pass 2: + competition features from pass-1 out-of-fold probabilities
          (who else is competing for this record / this entity, and by how much)

Precision-first: the model outputs calibrated probabilities; Stage 3 decides what to merge.
Folds are grouped by S1 entity, so no entity leaks between train and validation.
No country feature is used (France is unseen in training).

Steps (each resumable; models already on disk are reused):
  python src/stage2_matcher.py --work ../../work --suffix _s2 --step features --split train
  python src/stage2_matcher.py --work ../../work --suffix _s2 --step train
  python src/stage2_matcher.py --work ../../work --suffix _s2 --step features --split test
  python src/stage2_matcher.py --work ../../work --suffix _s2 --step predict
Full data: add --train-frac 0.3 to train on 30%% of S1 entities (RAM), no --suffix.
"""
import argparse
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling import in_train_frac, load_frame, save_parquet, shard_tag  # noqa: E402
import pair_features as pf  # noqa: E402

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
ID_COLS = ["s1_id", "x_id"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fold_of(ids, n):
    return np.array([zlib.crc32(e.encode()) % n for e in ids], np.int8)


# ------------------------------------------------------------------ eval ---
def macro_f05(gt_n: pd.Series, gt_pairs: set, pred: pd.DataFrame):
    """Official metric: per-S1 F0.5, singletons 1 if empty prediction else 0, macro mean.
    gt_n: number of true matches per S1 (index = every S1 being evaluated)."""
    if len(pred):
        hit = np.fromiter(((a, x) in gt_pairs for a, x in zip(pred.s1_id, pred.x_id)), bool, len(pred))
        g = pd.DataFrame({"s1": pred.s1_id.values, "tp": hit}).groupby("s1").tp.agg(["sum", "size"])
        tp = g["sum"].reindex(gt_n.index, fill_value=0).values
        npred = g["size"].reindex(gt_n.index, fill_value=0).values
    else:
        tp = npred = np.zeros(len(gt_n))
    nt = gt_n.values
    P = np.divide(tp, npred, out=np.zeros(len(nt)), where=npred > 0)
    Rr = np.divide(tp, nt, out=np.zeros(len(nt)), where=nt > 0)
    f = np.divide(1.25 * P * Rr, 0.25 * P + Rr, out=np.zeros(len(nt)), where=(0.25 * P + Rr) > 0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    single = nt == 0
    return f.mean(), f[single].mean() if single.any() else float("nan"), f[~single].mean()


def decide_threshold(c: pd.DataFrame, p: np.ndarray, t: float, one_owner=True):
    d = c[ID_COLS].assign(p=p)
    if one_owner:  # each S2/S3 record goes to its single best S1
        d = d[d.p == d.groupby("x_id").p.transform("max")]
    return d[d.p >= t]


def load_gt(args, s1_ids):
    path = (os.path.join(args.work, "norm", f"train_ground_truth{args.suffix}.tsv") if args.suffix
            else os.path.join(args.work, "..", "dataset", "train", "train_ground_truth.tsv"))
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1_ids)]
    pairs = {(a, x) for a, ids in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in ids.split(",") if x}
    n = gt.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    return n, pairs


# ------------------------------------------------------------- features ---
def load_stats(args):
    """Corpus statistics of a split, computed once (streaming) and cached."""
    import pickle
    nd = os.path.join(args.work, "norm")
    path = os.path.join(args.work, "stage2", f"stats_{args.split}{args.suffix}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            st = pickle.load(f)
    else:
        st = None
    if st is not None and "s1vocab" in st:
        return st
    if st is None:
        st = pf.build_stats_streaming(os.path.join(nd, f"{args.split}_source1{args.suffix}.parquet"),
                                      [os.path.join(nd, f"{args.split}_source{k}{args.suffix}.parquet") for k in (2, 3)])
    import json
    with open(os.path.join(args.work, "resources", "gazetteer.json"), encoding="utf-8") as fh:
        st["s1vocab"] = set(json.load(fh)["vocab"])       # name words seen in Source 1 (train+test)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(st, f)
    os.replace(path + ".tmp", path)
    return st


def step_features(args):
    nd, cd = os.path.join(args.work, "norm"), os.path.join(args.work, "cands")
    tag = shard_tag(args.shard, args.n_shards) if args.split == "test" else ""
    cands = pd.read_parquet(os.path.join(cd, f"{args.split}_final{args.suffix}{tag}.parquet"))
    order = np.argsort(cands.s1_id.values, kind="stable")      # keep each S1's candidates together
    cands = cands.iloc[order].reset_index(drop=True)
    if args.split == "train" and args.train_frac < 1:
        keep = np.array([in_train_frac(e, args.train_frac) for e in cands.s1_id])
        cands = cands[keep].reset_index(drop=True)
    st = load_stats(args)
    L = load_frame(os.path.join(nd, f"{args.split}_source1{args.suffix}.parquet"), pf.SIDE_COLS,
                   set(cands.s1_id.unique())).set_index("entity_id")
    idsx = set(cands.x_id.unique())
    R = pd.concat([load_frame(os.path.join(nd, f"{args.split}_source{k}{args.suffix}.parquet"), pf.SIDE_COLS, idsx)
                   for k in (2, 3)]).set_index("entity_id")
    log(f"{args.split}{tag}: {len(cands):,} pairs; stats ready")
    parts = []
    ids1 = cands.s1_id.values
    s = 0
    while s < len(cands):
        e = min(s + args.chunk, len(cands))
        while e < len(cands) and ids1[e] == ids1[e - 1]:   # never split an S1 entity across chunks
            e += 1
        parts.append(pf.compute(cands.iloc[s:e], L, R, st))
        log(f"  features {e:,}/{len(cands):,}")
        s = e
    X = pd.concat(parts, ignore_index=True)
    comp = pf.competition(cands, cands.prerank_p.values, "c0_")
    gx = os.path.join(args.work, "stage2", f"test_c0x{args.suffix}{tag}.parquet")
    if tag and os.path.exists(gx):   # sharded test: record-side competition over ALL shards
        g = pd.read_parquet(gx).iloc[order]            # global file is in the unsorted order
        for c in g.columns:
            comp[c] = g[c].values
    X = pd.concat([cands[ID_COLS].reset_index(drop=True), X, comp], axis=1)
    if args.split == "train":
        _, pairs = load_gt(args, L.index)
        X["y"] = np.fromiter(((a, x) in pairs for a, x in zip(X.s1_id, X.x_id)), bool, len(X)).astype(np.int8)
    dst = os.path.join(args.work, "stage2", f"{args.split}_feats{args.suffix}{tag}.parquet")
    save_parquet(X, dst)
    log(f"wrote {dst}: {X.shape}")


# ---------------------------------------------------------------- train ---
def _fit_or_load(path, Xtr, ytr, Xva, yva):
    if os.path.exists(path):
        return lgb.Booster(model_file=path)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), args_rounds[0], valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model(path, num_iteration=m.best_iteration)
    log(f"  saved {os.path.basename(path)} ({m.best_iteration} trees)")
    return m


args_rounds = [4000]


def step_train(args):
    sd, md = os.path.join(args.work, "stage2"), os.path.join(args.work, "models")
    X = pd.read_parquet(os.path.join(sd, f"train_feats{args.suffix}.parquet"))
    y = X.pop("y").values
    ids = X[ID_COLS]
    F1 = [c for c in X.columns if c not in ID_COLS]
    folds = fold_of(ids.s1_id.values, args.folds)
    log(f"train pairs {len(X):,}, positives {y.mean():.1%}, {len(F1)} features, {args.folds} folds")

    def run_pass(feats, tag):
        oof = np.zeros(len(X))
        for k in range(args.folds):
            tr, va = folds != k, folds == k
            m = _fit_or_load(os.path.join(md, f"stage2_{tag}_f{k}{args.suffix}.txt"),
                             X.loc[tr, feats], y[tr], X.loc[va, feats], y[va])
            oof[va] = m.predict(X.loc[va, feats])
        return oof

    p1 = run_pass(F1, "p1")
    comp = pf.competition(ids, p1, "m_")
    for c in comp.columns:
        X[c] = comp[c].values
    F2 = F1 + list(comp.columns)
    p2 = run_pass(F2, "p2")
    oof = ids.assign(y=y, p1=p1, p2=p2)
    save_parquet(oof, os.path.join(sd, f"train_oof{args.suffix}.parquet"))

    # ---- report
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"\nAUC  p1 {roc_auc_score(y, p1):.5f}  p2 {roc_auc_score(y, p2):.5f}")
    print(f"AP   p1 {average_precision_score(y, p1):.5f}  p2 {average_precision_score(y, p2):.5f}")
    s1_all = pd.read_parquet(os.path.join(args.work, "norm", f"train_source1{args.suffix}.parquet"), columns=["entity_id"]).entity_id
    if args.train_frac < 1:
        s1_all = s1_all[[in_train_frac(e, args.train_frac) for e in s1_all]]
    gt_n, gt_pairs = load_gt(args, s1_all)
    print(f"\nmacro F0.5 on {len(gt_n):,} train-sample S1 entities (OOF; includes pairs lost in blocking)")
    print("  threshold | pass-1 + one-owner | pass-2 + one-owner | pass-2, no one-owner   (all / singletons / matched)")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r1 = macro_f05(gt_n, gt_pairs, decide_threshold(ids, p1, t))
        r2 = macro_f05(gt_n, gt_pairs, decide_threshold(ids, p2, t))
        r3 = macro_f05(gt_n, gt_pairs, decide_threshold(ids, p2, t, one_owner=False))
        fmt = lambda r: f"{r[0]:.4f}/{r[1]:.3f}/{r[2]:.4f}"  # noqa: E731
        print(f"  {t:9.2f} | {fmt(r1)} | {fmt(r2)} | {fmt(r3)}")
    m = lgb.Booster(model_file=os.path.join(md, f"stage2_p2_f0{args.suffix}.txt"))
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    print("\ntop features (gain):", ", ".join(f"{k}" for k in imp.head(15).index))


# -------------------------------------------------------------- predict ---
def _avg(md, X, tag, args):
    ms = [lgb.Booster(model_file=os.path.join(md, f"stage2_{tag}_f{k}{args.suffix}.txt")) for k in range(args.folds)]
    return np.mean([mm.predict(X[mm.feature_name()]) for mm in ms], axis=0)


def step_predict(args):
    """predict = pass 1 + pass 2 in one go (unsharded).
    Sharded test: predict1 on every shard -> test_global.py --step m -> predict2 on every shard."""
    sd, md = os.path.join(args.work, "stage2"), os.path.join(args.work, "models")
    tag = shard_tag(args.shard, args.n_shards)
    X = pd.read_parquet(os.path.join(sd, f"test_feats{args.suffix}{tag}.parquet"))
    ids = X[ID_COLS]
    dst = os.path.join(sd, f"test_scores{args.suffix}{tag}.parquet")
    if args.step in ("predict", "predict1"):
        p1 = _avg(md, X, "p1", args)
        if args.step == "predict1":
            save_parquet(ids.assign(p1=p1), dst)
            log(f"wrote {dst} (pass 1): {len(ids):,} pairs")
            return
    else:
        p1 = pd.read_parquet(dst).p1.values
    comp = pf.competition(ids, p1, "m_")
    gx = os.path.join(sd, f"test_mx{args.suffix}{tag}.parquet")
    if tag and os.path.exists(gx):   # record-side competition over ALL shards
        g = pd.read_parquet(gx)
        for c in g.columns:
            comp[c] = g[c].values
    for c in comp.columns:
        X[c] = comp[c].values
    p2 = _avg(md, X, "p2", args)
    out = ids.assign(p1=p1, p2=p2)
    save_parquet(out, dst)
    log(f"wrote {dst}: {len(out):,} pairs; mean p2 {p2.mean():.3f}; pairs with p2>=0.5: {(p2 >= 0.5).sum():,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--step", choices=["features", "train", "predict", "predict1", "predict2"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--train-frac", type=float, default=1.0, help="share of train S1 entities used (RAM)")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    args_rounds[0] = args.rounds
    os.makedirs(os.path.join(args.work, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(args.work, "models"), exist_ok=True)
    {"features": step_features, "train": step_train, "predict": step_predict,
     "predict1": step_predict, "predict2": step_predict}[args.step](args)


if __name__ == "__main__":
    main()
