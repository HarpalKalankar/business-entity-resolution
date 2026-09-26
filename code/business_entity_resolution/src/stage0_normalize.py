"""Stage 0 entry point: build learned resources, then normalize every source file.

Usage (from code/business_entity_resolution):
    python src/stage0_normalize.py --data ../../dataset --work ../../work
    python src/stage0_normalize.py --data ../../dataset --work ../../work --sample-pct 5   # dev subset

Outputs (in --work):
    resources/translit.json    learned native-script -> Latin tables
    resources/gazetteer.json   per-country cities / states + name vocab
    norm/{split}_source{k}[ _sN].parquet   one row per record, normalized fields
    norm/train_ground_truth[_sN].tsv       (sample mode) ground truth restricted to sampled S1
"""
import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool, cpu_count

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gazetteer as gz  # noqa: E402
import translit_dict  # noqa: E402
from address_norm import AddressNormalizer  # noqa: E402
from name_norm import NameNormalizer, Segmenter  # noqa: E402

COLUMNS = ["entity_id", "src", "country", "name_raw", "addr_raw",
           "name_clean", "name_core", "name_key", "name_alt", "legal", "name_translit", "name_is_domain",
           "addr_clean", "house_num", "all_nums", "street", "locality", "city", "state", "postcode",
           "unit", "landmark", "addr_missing", "addr_translit"]
INT_COLS = {"src", "name_translit", "name_is_domain", "addr_missing", "addr_translit"}

_W = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _init_worker(translit_path, gaz_path):
    tr = translit_dict.Transliterator.load(translit_path)
    g = gz.load(gaz_path)
    _W["name"] = NameNormalizer(tr, Segmenter(g["vocab"]))
    _W["addr"] = AddressNormalizer(tr, g["countries"])


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    nn, an = _W["name"], _W["addr"]
    rows = []
    for eid, name, addr, country in df[["entity_id", "business_name", "business_address", "country"]].itertuples(index=False):
        r = {"entity_id": eid, "src": int(eid[1]) if len(eid) > 1 and eid[1].isdigit() else 0,
             "country": (country or "").strip(), "name_raw": name, "addr_raw": addr}
        r.update(nn.normalize(name))
        r.update(an.normalize(addr, country))
        rows.append(r)
    out = pd.DataFrame(rows, columns=COLUMNS)
    for c in INT_COLS:
        out[c] = out[c].astype("int8")
    return out


def _work(df):
    return normalize_frame(df)


def keep_id(eid: str, pct: float) -> bool:
    return (zlib.crc32(eid.encode()) % 10000) < pct * 100


def read_chunks(path, chunksize, id_filter=None):
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                             chunksize=chunksize, encoding="utf-8"):
        if id_filter is not None:
            chunk = chunk[chunk["entity_id"].map(id_filter)]
        if len(chunk):
            yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder containing train/ and test/")
    ap.add_argument("--work", required=True, help="output folder for resources and parquet files")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--sample-pct", type=float, default=100.0,
                    help="keep this %% of S1 entities (+ all their train matches, + the same %% of other S2/S3)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--rebuild", action="store_true", help="re-learn resources even if they exist")
    ap.add_argument("--resources-only", action="store_true", help="only build translit/gazetteer resources")
    ap.add_argument("--only", nargs="*", default=None,
                    help="process only these files, e.g. train_source1 test_source2")
    ap.add_argument("--overwrite", action="store_true", help="re-normalize files whose parquet exists")
    args = ap.parse_args()

    res_dir = os.path.join(args.work, "resources")
    norm_dir = os.path.join(args.work, "norm")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    tr_path = os.path.join(res_dir, "translit.json")
    gz_path = os.path.join(res_dir, "gazetteer.json")

    if args.rebuild or not os.path.exists(tr_path):
        log("learning transliteration tables from training ground truth ...")
        translit_dict.build(args.data, tr_path, log=lambda m: log(m))
    if args.rebuild or not os.path.exists(gz_path):
        log("mining gazetteer + vocab from Source 1 ...")
        gz.build(args.data, translit_dict.Transliterator.load(tr_path), gz_path, log=lambda m: log(m))

    if args.resources_only:
        log("resources ready")
        return

    pct = args.sample_pct
    suffix = "" if pct >= 100 else f"_s{pct:g}"
    linked = set()
    wants = lambda name: args.only is None or name in args.only  # noqa: E731
    if pct < 100 and "train" in args.splits and (wants("train_source2") or wants("train_source3")):
        # sample mode: keep every train S2/S3 record linked to a sampled S1 (+ pct% of the rest)
        gt_in = os.path.join(args.data, "train", "train_ground_truth.tsv")
        gt_out = os.path.join(norm_dir, f"train_ground_truth{suffix}.tsv")
        with open(gt_in, encoding="utf-8") as f, open(gt_out, "w", encoding="utf-8") as g:
            g.write(next(f))
            for line in f:
                s1, _, ids = line.rstrip("\n").partition("\t")
                if keep_id(s1, pct):
                    g.write(line)
                    linked.update(x for x in ids.split(",") if x)
        log(f"sample {pct}%: ground truth written, {len(linked):,} linked S2/S3 ids kept")

    with Pool(args.workers, initializer=_init_worker, initargs=(tr_path, gz_path)) as pool:
        for split in args.splits:
            for k in (1, 2, 3):
                src = os.path.join(args.data, split, f"{split}_source{k}.tsv")
                dst = os.path.join(norm_dir, f"{split}_source{k}{suffix}.parquet")
                if args.only and f"{split}_source{k}" not in args.only:
                    continue
                if os.path.exists(dst) and not args.overwrite:
                    log(f"skip {dst} (exists; use --overwrite)")
                    continue
                if pct >= 100:
                    flt = None
                elif k == 1:
                    flt = lambda e: keep_id(e, pct)  # noqa: E731
                else:
                    flt = lambda e: e in linked or keep_id(e, pct)  # noqa: E731
                t0, n, writer = time.time(), 0, None
                for out in pool.imap(_work, read_chunks(src, args.chunksize, flt)):
                    table = pa.Table.from_pandas(out, preserve_index=False)
                    if writer is None:
                        tmp = dst + ".part"
                        writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                    writer.write_table(table)
                    n += len(out)
                    log(f"  {split}_source{k}: {n:,} rows ({n / (time.time() - t0):,.0f} rows/s)")
                if writer is not None:
                    writer.close()
                    os.replace(tmp, dst)
                log(f"done {dst}  rows={n:,}  {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
