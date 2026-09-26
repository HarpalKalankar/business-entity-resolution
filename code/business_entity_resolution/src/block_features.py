"""Blocking 'signatures' for Stage 1.

Every record becomes a small bag of hashed features. Two records become candidates when
they share rare features; the candidate score is the sum of IDF weights of the shared ones.

Why conjunctions: names in this dataset reuse a small vocabulary ('private', 'marketing',
'global' ...), so single name tokens are too common to block on. Pairing a name token
with the street, city or house number yields rare, robust keys:
    'kayleys|lake'  'kayleys|troy'  '27|kayleys'  ...
Frequent features (document frequency above a cap) are dropped, which bounds the fan-out.
"""
import numpy as np
import pandas as pd

from address_norm import STREET_TYPES
from text_utils import phonetic_squeeze

# feature family -> weight multiplier on top of IDF
W = {
    "k": 2.0,    # exact name_key
    "t": 1.0,    # name token
    "b": 1.0,    # adjacent name-token bigram
    "p": 0.6,    # phonetic token (typos, transliteration variants)
    "ts": 1.0,   # name token x street word
    "tc": 1.0,   # name token x city
    "tst": 0.5,  # name token x state
    "th": 1.0,   # name token x house number
    "hs": 1.0,   # house number x street word
    "hc": 0.8,   # house number x city
    # address-only keys: catch trade names / brand names at the same address
    "ae": 1.0,   # whole normalized address
    "sb": 0.7,   # street-word bigram x city
    "hl": 0.7,   # house number x locality
    "ul": 0.7,   # unit x street word / city
    # acronyms: 'MC' for 'Mohindra & Co', 'HI' for 'Harsha Info'
    "ih": 0.8,   # initials x house number
    "ic": 0.6,   # initials x city
}
_STOP = {"and", "of", "the", "et", "de", "du", "des", "la", "le"}
MAX_NAME_TOK = 5


def _name_tokens(core: str, alt: str):
    toks = [t for t in core.split() if len(t) > 1 and t not in _STOP]
    if alt:
        for a in alt.split(" | "):
            toks += [t for t in a.split() if len(t) > 1 and t not in _STOP and t not in toks]
    return toks[:MAX_NAME_TOK + 2]


def _initials(clean: str, core: str):
    out = set()
    for s in (clean, core):
        t = [w for w in s.split() if w not in _STOP]
        if len(t) >= 2:
            out.add("".join(w[0] for w in t))
    ct = core.split()
    if len(ct) == 1 and 2 <= len(ct[0]) <= 5 and ct[0].isalpha():
        out.add(ct[0])  # the name itself may be an acronym
    return out


def record_features(core, key, alt, street, city, state, house, clean="", locality="", unit="", addr=""):
    """Return list of (feature_string, family)."""
    out = []
    toks = _name_tokens(core, alt)
    if key:
        out.append(("k:" + key, "k"))
    for t in toks:
        out.append(("t:" + t, "t"))
        if len(t) >= 4:
            out.append(("p:" + phonetic_squeeze(t), "p"))
    ct = core.split()
    for a, b in zip(ct, ct[1:]):
        out.append(("b:" + a + "_" + b, "b"))
    sw = [w for w in street.split() if w not in STREET_TYPES and not w.isdigit() and len(w) > 1][:2]
    st = state.split(" | ")[0] if state else ""
    for t in toks[:MAX_NAME_TOK]:
        for w in sw:
            out.append((f"ts:{t}|{w}", "ts"))
        if city:
            out.append((f"tc:{t}|{city}", "tc"))
        if st:
            out.append((f"tst:{t}|{st}", "tst"))
        if house:
            out.append((f"th:{t}|{house}", "th"))
    if house:
        for w in sw:
            out.append((f"hs:{house}|{w}", "hs"))
        if city:
            out.append((f"hc:{house}|{city}", "hc"))
        for loc in locality.split(" | ")[:2]:
            if loc:
                out.append((f"hl:{house}|{loc}", "hl"))
    if addr and len(addr) >= 12:
        out.append(("ae:" + addr, "ae"))
    swa = [w for w in street.split() if not w.isdigit()]
    for a, b in zip(swa, swa[1:]):
        out.append((f"sb:{a}_{b}|{city}", "sb"))
    if unit:
        u = unit.split()[0]
        for w in sw[:1]:
            out.append((f"ul:{u}|{w}", "ul"))
        if city:
            out.append((f"ul:{u}|{city}", "ul"))
    for ini in _initials(clean, core):
        if house:
            out.append((f"ih:{ini}|{house}", "ih"))
        if city:
            out.append((f"ic:{ini}|{city}", "ic"))
    return out


FAMILIES = list(W)
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
FAM_W = np.array([W[f] for f in FAMILIES], dtype=np.float32)


def frame_features(df: pd.DataFrame, chunk: int = 200_000):
    """Vectorise a normalized frame -> (row index, feature hash uint64, family id).
    Processed in chunks so Python strings never pile up (millions of rows on the full data)."""
    R, H, F = [], [], []
    cols = ["name_core", "name_key", "name_alt", "street", "city", "state", "house_num",
            "name_clean", "locality", "unit", "addr_clean"]
    for s in range(0, len(df), chunk):
        rows, feats, fams = [], [], []
        for i, row in enumerate(df[cols].iloc[s:s + chunk].itertuples(index=False), start=s):
            for f, fam in record_features(*row):
                rows.append(i)
                feats.append(f)
                fams.append(FAM_ID[fam])
        if feats:
            R.append(np.asarray(rows, np.int64))
            H.append(pd.util.hash_array(np.asarray(feats, dtype=object)))
            F.append(np.asarray(fams, np.int8))
    if not R:
        return (np.zeros(0, np.int64), np.zeros(0, np.uint64), np.zeros(0, np.int8))
    return np.concatenate(R), np.concatenate(H), np.concatenate(F)
