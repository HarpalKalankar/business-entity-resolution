"""End-to-end runner: data -> normalization -> blocking -> matching -> output/*.tsv -> validator.

Full run on a 16 GB PC (writes output/candidate_pairs.tsv and output/matching_results.tsv):
    python src/run_pipeline.py
Dev run on a 2% sample (writes *_s2.tsv, no validator):
    python src/run_pipeline.py --sample-pct 2
Resume:  --from stage1 | stage2 | stage3 | test   (finished test shard steps are always skipped;
         delete work/cands/*.done and the shard files to force a test re-run)

Memory plan
  * training uses --train-frac of the train S1 entities (default 0.10), blocked against the
    FULL train S2/S3 pools so name collisions are realistic
  * the test set is processed in --shards hash shards of S1 entities (default 8); each shard
    runs blocking -> pre-ranker -> features -> prediction -> decision, then stage3_merge
    joins them and applies the one-owner rule across shards
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["stage0", "stage1", "stage2", "stage3", "test"]


def run(script, *a):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, a)]
    print("\n>>", os.path.basename(script), " ".join(map(str, a)), flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # Windows console: Indic names
    subprocess.run(cmd, check=True, env=env)
    print(f"   ({(time.time() - t0) / 60:,.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../dataset")
    ap.add_argument("--work", default="../../work")
    ap.add_argument("--out", default="../../output")
    ap.add_argument("--sample-pct", type=float, default=100.0)
    ap.add_argument("--train-frac", type=float, default=0.10, help="share of train S1 entities used for training")
    ap.add_argument("--shards", type=int, default=None, help="test S1 shards (default 8 full, 1 sample)")
    ap.add_argument("--k", type=int, default=15, help="final candidates per S1 entity (smaller ranks higher)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from", dest="start", default="stage0", choices=STAGES)
    args = ap.parse_args()
    sfx = "" if args.sample_pct >= 100 else f"_s{args.sample_pct:g}"
    tf = args.train_frac if not sfx else 1.0          # the sample is already small
    n = args.shards or (1 if sfx else 8)
    W, S = ["--work", args.work], (["--suffix", sfx] if sfx else [])
    todo = STAGES[STAGES.index(args.start):]
    T0 = time.time()

    # ---------------- training side
    if "stage0" in todo:
        run("stage0_normalize.py", "--data", args.data, *W, "--sample-pct", args.sample_pct, "--workers", args.workers)
    if "stage1" in todo:
        run("stage1_blocking.py", *W, *S, "--split", "train", "--train-frac", tf)
        run("stage1_eval.py", *W, *S, "--train-frac", tf)
        run("stage1_prerank.py", *W, *S, "--mode", "train", "--train-frac", tf)
        run("stage1_prerank.py", *W, *S, "--mode", "apply", "--split", "train", "--k", args.k)
    if "stage2" in todo:
        run("stage2_matcher.py", *W, *S, "--step", "features", "--split", "train")
        run("stage2_matcher.py", *W, *S, "--step", "train", "--train-frac", tf)
    if "stage3" in todo:
        run("stage3_decide.py", *W, *S, "--mode", "tune", "--train-frac", tf)

    # ---------------- test side
    if n == 1:
        run("stage1_blocking.py", *W, *S, "--split", "test")
        run("stage1_prerank.py", *W, *S, "--mode", "apply", "--split", "test", "--k", args.k, "--out", args.out)
        run("stage2_matcher.py", *W, *S, "--step", "features", "--split", "test")
        run("stage2_matcher.py", *W, *S, "--step", "predict")
        run("stage3_decide.py", *W, *S, "--mode", "apply", "--out", args.out)
    else:
        # Sharded: every pass that needs a record's competitors across ALL S1 entities is done
        # globally by test_global.py on compact numeric arrays, so sharded == unsharded.
        G = [*W, *S, "--n-shards", n]
        sh = lambda i: ["--shard", i, "--n-shards", n]  # noqa: E731
        tagf = lambda kind, i: os.path.join(args.work, kind, f"{sfx}_sh{i}of{n}")  # noqa: E731

        def per_shard(label, fn, marker):
            for i in range(n):
                if os.path.exists(marker(i)):
                    print(f"   {label} shard {i + 1}/{n}: done, skipping")
                    continue
                print(f"\n== {label}: shard {i + 1}/{n}")
                fn(i)

        mk = lambda d, name: (lambda i: os.path.join(args.work, d, f"{name}{sfx}_sh{i}of{n}.parquet"))  # noqa: E731
        flag = lambda name: os.path.join(args.work, "cands", f"{name}{sfx}_of{n}.done")  # noqa: E731
        per_shard("blocking", lambda i: run("stage1_blocking.py", *W, *S, "--split", "test", *sh(i)),
                  mk("cands", "test_cands"))
        if not os.path.exists(flag("revrank")):
            run("test_global.py", *G, "--step", "revrank")
            open(flag("revrank"), "w").close()
        per_shard("pre-ranker", lambda i: run("stage1_prerank.py", *W, *S, "--mode", "apply", "--split", "test",
                                              "--k", args.k, "--out", args.out, *sh(i)), mk("cands", "test_final"))
        if not os.path.exists(flag("c0")):
            run("test_global.py", *G, "--step", "c0")
            open(flag("c0"), "w").close()
        per_shard("features", lambda i: run("stage2_matcher.py", *W, *S, "--step", "features", "--split", "test",
                                            *sh(i)), mk("stage2", "test_feats"))
        per_shard("pass 1", lambda i: run("stage2_matcher.py", *W, *S, "--step", "predict1", *sh(i)),
                  mk("stage2", "test_scores"))
        if not os.path.exists(flag("m")):
            run("test_global.py", *G, "--step", "m")
            open(flag("m"), "w").close()
        per_shard("pass 2 + decision", lambda i: (run("stage2_matcher.py", *W, *S, "--step", "predict2", *sh(i)),
                                                  run("stage3_decide.py", *W, *S, "--mode", "apply", *sh(i))),
                  mk("stage3", "test_matches"))
        run("stage3_merge.py", *W, *S, "--n-shards", n, "--out", args.out)
    print(f"\nTOTAL {(time.time() - T0) / 3600:.2f} h")

    if not sfx:
        root = os.path.abspath(os.path.join(args.data, ".."))
        subprocess.run([sys.executable, os.path.join(root, "utils", "validate_submission.py"),
                        "--matching", os.path.join(args.out, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data, "test")], check=False)


if __name__ == "__main__":
    main()
