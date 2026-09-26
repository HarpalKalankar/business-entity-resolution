"""Stage 2 pair features (~55 numeric columns per (S1, candidate) pair).

Deliberately country-agnostic: no country column, only similarity / agreement signals,
so the model transfers to France, which has no labels.
Groups:
  name      fuzzy scores, IDF-weighted soft overlap, *unmatched* rare tokens (what makes a
            look-alike a different business), phonetic, legal form, aliases
  address   house number (exact / suffix / numeric distance / in other's numbers), street,
            city, state, postcode, unit, locality, whole address
  common    how many records share this name key / house+street (a common name needs an address)
  context   blocking score & ranks, pre-ranker probability, competition among candidates
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from text_utils import phonetic_squeeze

SIDE_COLS = ["entity_id", "country", "name_clean", "name_core", "name_key", "name_alt", "legal", "name_translit",
             "name_is_domain", "addr_clean", "house_num", "all_nums", "street", "locality", "city", "state",
             "postcode", "unit", "landmark", "addr_missing"]


# ------------------------------------------------------------------ stats ---
def build_stats_streaming(s1_path, x_paths, batch=500_000) -> dict:
    """Same statistics as build_stats, read in batches from parquet (full data fits in RAM)."""
    import pyarrow.parquet as pq
    df, n = Counter(), 0
    key_s1, key_x, hs_x = Counter(), Counter(), Counter()
    cols = ["country", "name_core", "name_key", "house_num", "street"]
    for path, is_s1 in [(s1_path, True)] + [(p, False) for p in x_paths]:
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=cols):
            d = b.to_pandas()
            n += len(d)
            for s in d.name_core.values:
                df.update(set(s.split()))
            keys = (d.country + "|" + d.name_key).values
            if is_s1:
                key_s1.update(keys)
            else:
                key_x.update(keys)
                m = d.house_num.values != ""
                hs_x.update((d.house_num.values[m] + "|" + d.street.values[m]))
    idf = {t: math.log(n / c) for t, c in df.items()}
    return {"idf": idf, "idf_max": math.log(n), "key_s1": pd.Series(key_s1, dtype=float),
            "key_x": pd.Series(key_x, dtype=float), "hs_x": pd.Series(hs_x, dtype=float)}


def build_stats(L: pd.DataFrame, R: pd.DataFrame) -> dict:
    """Token IDF and commonness counts, computed on the split being scored (no labels used)."""
    df = Counter()
    for s in pd.concat([L.name_core, R.name_core]).values:
        df.update(set(s.split()))
    n = len(L) + len(R)
    idf = {t: math.log(n / c) for t, c in df.items()}
    key_s1 = (L.country + "|" + L.name_key).value_counts()
    key_x = (R.country + "|" + R.name_key).value_counts()
    hs = R.house_num + "|" + R.street
    hs_x = hs[R.house_num != ""].value_counts()
    return {"idf": idf, "idf_max": math.log(n), "key_s1": key_s1, "key_x": key_x, "hs_x": hs_x}


# -------------------------------------------------------------- helpers ---
def _jacc(a, b):
    if not a or not b:
        return -1.0
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b)


def _eq(a, b):
    """1 equal, 0 different, -1 either missing."""
    a, b = np.asarray(a, object), np.asarray(b, object)
    miss = (a == "") | (b == "")
    return np.where(miss, -1, (a == b).astype(np.int8)).astype(np.int8)


def _soft_idf(a: str, b: str, idf: dict, dflt: float):
    """IDF-weighted soft token overlap. Returns (cov_a, cov_b, unmatched_a, unmatched_b, shared)."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return -1.0, -1.0, -1.0, -1.0, 0.0
    sb, sa = set(tb), set(ta)

    def side(x, yset, ylist):
        tot = mat = 0.0
        for t in set(x):
            w = idf.get(t, dflt)
            tot += w
            if t in yset or any(JaroWinkler.normalized_similarity(t, u) >= 0.9 for u in ylist if abs(len(t) - len(u)) <= 2):
                mat += w
        return tot, mat

    tot_a, mat_a = side(ta, sb, tb)
    tot_b, mat_b = side(tb, sa, ta)
    return (mat_a / tot_a if tot_a else 0.0, mat_b / tot_b if tot_b else 0.0,
            tot_a - mat_a, tot_b - mat_b, sum(idf.get(t, dflt) for t in sa & sb))


def _house_feats(ha, hb, na, nb):
    out = np.zeros((len(ha), 4), np.float32)
    for i, (a, b, xa, xb) in enumerate(zip(ha, hb, na, nb)):
        if not a or not b:
            out[i] = (-1, -1, -1, 1 if (xa and a and a in xb.split()) or (xb and b and b in xa.split()) else 0)
            continue
        suffix = int(a != b and (a.endswith(b) or b.endswith(a)))
        try:
            diff = math.log1p(abs(int(a) - int(b)))
        except ValueError:
            diff = -1
        cross = int(a in xb.split() or b in xa.split())
        out[i] = (suffix, diff, cross, cross)
    return out


# ------------------------------------------------------------- features ---
def compute(c: pd.DataFrame, L: pd.DataFrame, R: pd.DataFrame, st: dict) -> pd.DataFrame:
    """c: candidate pairs with s1_id, x_id, src, score, fwd_rank, rev_rank, prerank_p.
    L / R: normalized S1 / S2+S3 frames indexed by entity_id."""
    li, ri = L.index.get_indexer(c.s1_id), R.index.get_indexer(c.x_id)
    A = {k: L[k].values[li] for k in SIDE_COLS[1:]}
    B = {k: R[k].values[ri] for k in SIDE_COLS[1:]}
    f = {}
    # --- name: fuzzy (vectorised C++)
    f["n_ratio"] = cpdist(A["name_core"], B["name_core"], scorer=fuzz.ratio, workers=-1)
    f["n_tsr"] = cpdist(A["name_core"], B["name_core"], scorer=fuzz.token_set_ratio, workers=-1)
    f["n_tsort"] = cpdist(A["name_core"], B["name_core"], scorer=fuzz.token_sort_ratio, workers=-1)
    f["n_partial"] = cpdist(A["name_core"], B["name_core"], scorer=fuzz.partial_ratio, workers=-1)
    f["n_key_jw"] = cpdist(A["name_key"], B["name_key"], scorer=JaroWinkler.normalized_similarity, workers=-1)
    f["n_key_lev"] = cpdist(A["name_key"], B["name_key"], scorer=Levenshtein.normalized_similarity, workers=-1)
    f["n_clean_tsr"] = cpdist(A["name_clean"], B["name_clean"], scorer=fuzz.token_set_ratio, workers=-1)
    f["n_key_eq"] = (A["name_key"] == B["name_key"]).astype(np.int8)
    f["n_core_eq"] = (A["name_core"] == B["name_core"]).astype(np.int8)
    f["n_first_eq"] = np.array([a.split()[:1] == b.split()[:1] for a, b in zip(A["name_core"], B["name_core"])], np.int8)
    # aliases (dba / formerly known as): best of cross comparisons
    alt = np.zeros(len(c), np.float32)
    has_alt = (A["name_alt"] != "") | (B["name_alt"] != "")
    if has_alt.any():
        idx = np.where(has_alt)[0]
        s1 = cpdist(A["name_alt"][idx], B["name_core"][idx], scorer=fuzz.token_set_ratio, workers=-1)
        s2 = cpdist(A["name_core"][idx], B["name_alt"][idx], scorer=fuzz.token_set_ratio, workers=-1)
        alt[idx] = np.maximum(s1, s2)
    f["n_alt_tsr"] = alt
    # IDF soft overlap and unmatched rare tokens
    idf, dflt = st["idf"], st["idf_max"]
    soft = np.array([_soft_idf(a, b, idf, dflt) for a, b in zip(A["name_core"], B["name_core"])], np.float32)
    f["n_cov_a"], f["n_cov_b"], f["n_unm_a"], f["n_unm_b"], f["n_shared_idf"] = soft.T
    f["n_unm_max"] = np.maximum(soft[:, 2], soft[:, 3])
    f["n_phon_jacc"] = [_jacc(" ".join(phonetic_squeeze(t) for t in a.split()), " ".join(phonetic_squeeze(t) for t in b.split()))
                        for a, b in zip(A["name_core"], B["name_core"])]
    f["n_len_a"] = [len(s.split()) for s in A["name_core"]]
    f["n_len_b"] = [len(s.split()) for s in B["name_core"]]
    f["legal_eq"] = _eq(A["legal"], B["legal"])
    f["legal_overlap"] = [_jacc(a, b) for a, b in zip(A["legal"], B["legal"])]
    f["translit_a"], f["translit_b"] = A["name_translit"], B["name_translit"]
    f["domain_b"] = B["name_is_domain"]
    # --- address
    f["h_eq"] = _eq(A["house_num"], B["house_num"])
    hf = _house_feats(A["house_num"], B["house_num"], A["all_nums"], B["all_nums"])
    f["h_suffix"], f["h_logdiff"], f["h_cross"], f["h_in_nums"] = hf.T
    f["nums_jacc"] = [_jacc(a, b) for a, b in zip(A["all_nums"], B["all_nums"])]
    f["st_tsr"] = cpdist(A["street"], B["street"], scorer=fuzz.token_set_ratio, workers=-1)
    f["st_jw"] = cpdist(A["street"], B["street"], scorer=JaroWinkler.normalized_similarity, workers=-1)
    f["st_jacc"] = [_jacc(a, b) for a, b in zip(A["street"], B["street"])]
    f["st_missing"] = ((A["street"] == "").astype(np.int8) + (B["street"] == "").astype(np.int8))
    f["city_eq"] = _eq(A["city"], B["city"])
    f["city_jw"] = cpdist(A["city"], B["city"], scorer=JaroWinkler.normalized_similarity, workers=-1)
    f["state_eq"] = _eq(A["state"], B["state"])
    f["pc_eq"] = _eq(A["postcode"], B["postcode"])
    f["unit_eq"] = _eq(A["unit"], B["unit"])
    f["loc_tsr"] = cpdist(A["locality"], B["locality"], scorer=fuzz.token_set_ratio, workers=-1)
    f["addr_tsr"] = cpdist(A["addr_clean"], B["addr_clean"], scorer=fuzz.token_set_ratio, workers=-1)
    f["addr_ratio"] = cpdist(A["addr_clean"], B["addr_clean"], scorer=fuzz.ratio, workers=-1)
    f["addr_jacc"] = [_jacc(a, b) for a, b in zip(A["addr_clean"], B["addr_clean"])]
    f["addr_missing_b"] = B["addr_missing"]
    f["landmark_b"] = (B["landmark"] != "").astype(np.int8)
    # --- commonness (log counts; a common name must be backed by the address)
    ka = pd.Series(A["country"] + "|" + A["name_key"])
    f["key_cnt_s1"] = np.log1p(ka.map(st["key_s1"]).fillna(0).values)
    f["key_cnt_x"] = np.log1p(ka.map(st["key_x"]).fillna(0).values)
    hsb = pd.Series(B["house_num"] + "|" + B["street"])
    f["hs_cnt_x"] = np.log1p(hsb.map(st["hs_x"]).fillna(0).values)
    # --- trade / brand names: share of name words never seen in any Source 1 name
    #     ('Korectodelta' at the S1's exact address is a DBA name -> judge by the address)
    voc = st.get("s1vocab")
    if voc is not None:
        def oov(s):
            t = [w for w in s.split() if len(w) >= 4 and not w.isdigit()]
            return sum(w not in voc for w in t) / len(t) if t else -1.0
        f["b_oov"] = [oov(s) for s in B["name_core"]]
        f["a_oov"] = [oov(s) for s in A["name_core"]]
    f["b_one_tok"] = np.array([len(s.split()) == 1 for s in B["name_core"]], np.int8)
    f["addr_eq"] = np.where((A["addr_clean"] == "") | (B["addr_clean"] == ""), -1,
                            (A["addr_clean"] == B["addr_clean"]).astype(np.int8)).astype(np.int8)
    # --- siblings: similarity of this record to the S1 entity's OTHER confident candidates
    #     (a no-address 'Peak Obic' next to a confident 'Peak Obic' with full address)
    f.update(_siblings(c, B))
    # --- blocking / pre-ranker context
    for k in ("src", "score", "fwd_rank", "rev_rank", "prerank_p"):
        f[k] = c[k].values
    out = pd.DataFrame(f)
    for k in out.columns:
        if out[k].dtype == np.float64:
            out[k] = out[k].astype(np.float32)
    return out


def _siblings(c: pd.DataFrame, B: dict, conf: float = 0.9) -> dict:
    """Needs every candidate of an S1 entity in the same chunk (the caller aligns chunks)."""
    n = len(c)
    g = pd.factorize(c.s1_id.values)[0]
    sib = np.flatnonzero(c.prerank_p.values >= conf)
    out = {"sib_n": np.zeros(n, np.float32), "sib_name_max": np.full(n, -1.0, np.float32),
           "sib_addr_max": np.full(n, -1.0, np.float32), "sib_key_eq": np.zeros(n, np.int8)}
    if len(sib) == 0:
        return out
    L = pd.DataFrame({"r": np.arange(n), "g": g})
    Rt = pd.DataFrame({"j": sib, "g": g[sib]})
    m = L.merge(Rt, on="g")
    m = m[m.r != m.j]
    if len(m) == 0:
        return out
    r, j = m.r.values, m.j.values
    nm = cpdist(B["name_core"][r], B["name_core"][j], scorer=fuzz.token_set_ratio, workers=-1)
    ad = cpdist(B["addr_clean"][r], B["addr_clean"][j], scorer=fuzz.token_set_ratio, workers=-1)
    ke = (B["name_key"][r] == B["name_key"][j]) & (B["name_key"][r] != "")
    agg = pd.DataFrame({"r": r, "nm": nm, "ad": ad, "ke": ke}).groupby("r").agg(
        n=("nm", "size"), nm=("nm", "max"), ad=("ad", "max"), ke=("ke", "max"))
    out["sib_n"][agg.index.values] = agg.n.values
    out["sib_name_max"][agg.index.values] = agg.nm.values
    out["sib_addr_max"][agg.index.values] = agg.ad.values
    out["sib_key_eq"][agg.index.values] = agg.ke.values.astype(np.int8)
    return out


def competition(c: pd.DataFrame, p: np.ndarray, prefix: str) -> pd.DataFrame:
    """Competition among candidates, from probabilities p (pre-ranker or pass-1 model).
    Uses the fact that each S2/S3 record belongs to at most one S1 entity."""
    d = pd.DataFrame({"s1": c.s1_id.values, "x": c.x_id.values, "p": p})
    g1, gx = d.groupby("s1").p, d.groupby("x").p
    out = pd.DataFrame(index=d.index)
    out[f"{prefix}rank_s1"] = g1.rank(ascending=False, method="min").astype(np.float32)
    out[f"{prefix}gap_s1"] = (g1.transform("max") - d.p).astype(np.float32)
    out[f"{prefix}n_hi_s1"] = d.assign(h=d.p > 0.5).groupby("s1").h.transform("sum").astype(np.float32)
    out[f"{prefix}sum_s1"] = g1.transform("sum").astype(np.float32)
    out[f"{prefix}rank_x"] = gx.rank(ascending=False, method="min").astype(np.float32)
    out[f"{prefix}gap_x"] = (gx.transform("max") - d.p).astype(np.float32)
    out[f"{prefix}n_x"] = gx.transform("size").astype(np.float32)
    s = d.sort_values(["x", "p"], ascending=[True, False])
    s["r"] = s.groupby("x").cumcount()
    second = d.x.map(s[s.r == 1].set_index("x").p).fillna(0.0).values
    out[f"{prefix}margin_x"] = (d.p - np.where(out[f"{prefix}rank_x"] == 1, second, gx.transform("max"))).astype(np.float32)
    return out
