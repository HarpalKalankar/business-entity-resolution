"""Deterministic entity subsets + memory-friendly loading.

All subsets use salted CRC32 hashes, so train fraction, shards and the stage-0 sample
are independent of each other.
"""
import zlib

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def in_train_frac(entity_id: str, frac: float) -> bool:
    return frac >= 1 or zlib.crc32(("trainfrac|" + entity_id).encode()) % 1000 < frac * 1000


def in_shard(entity_id: str, shard: int, n_shards: int) -> bool:
    return n_shards <= 1 or zlib.crc32(("shard|" + entity_id).encode()) % n_shards == shard


def shard_tag(shard: int, n_shards: int) -> str:
    return "" if n_shards <= 1 else f"_sh{shard}of{n_shards}"


def load_frame(path, columns, ids=None, country=None):
    """Read a normalized parquet keeping strings in compact Arrow form until the rows are
    filtered, then convert only what is needed to pandas (big RAM saving on the full data)."""
    t = pq.read_table(path, columns=columns)
    if ids is not None:
        t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(list(ids), pa.string())))
    if country is not None:
        t = t.filter(pc.equal(t["country"], country))
    return t.to_pandas()


def save_parquet(df, path):
    """Atomic write: a crash or Ctrl+C never leaves a half-written file that a resume would trust."""
    import os
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
