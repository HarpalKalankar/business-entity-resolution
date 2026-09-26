"""Mine a per-country gazetteer (cities, states/regions) and a name-token vocabulary
from Source 1 (the clean reference source), train + test. Nothing external is used.

 * cities  : alphabetic comma-parts seen >= MIN_CITY times in a country
 * states  : frequent parts that usually sit in the LAST position (works for unseen
             countries such as France: 'nouvelle aquitaine', 'hauts de france', 'gironde')
 * vocab   : name-token counts, used to segment glued domain names
"""
import json
import os
import re
from collections import Counter, defaultdict

from address_norm import STREET_TYPES, build_state_lookup, is_french, norm_part_text
from text_utils import has_nonlatin, to_ascii

MIN_CITY = 3
MIN_STATE = 30
LAST_SHARE = 0.5
_NUM = re.compile(r"\d")
_NOT_PLACE = {"floor", "building", "bldg", "wing", "block", "tower", "complex", "apartment", "apartments",
              "plot", "flat", "shop", "house", "room", "unit", "suite", "office", "chaussee", "rdc",
              "maison", "mall", "market", "hotel", "hospital", "school", "near", "opposite", "behind"}


def _iter_s1(data_dir):
    for split in ("train", "test"):
        path = f"{data_dir}/{split}/{split}_source1.tsv"
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                p += [""] * (4 - len(p))
                yield p


def build(data_dir, translit, out_path, log=print):
    memo = {}

    def norm(p, fr):
        k = (p, fr)
        v = memo.get(k)
        if v is None:
            v = norm_part_text(translit.part(p) if has_nonlatin(p) else p, fr)
            if len(memo) < 3_000_000:
                memo[k] = v
        return v

    cnt = defaultdict(Counter)
    last = defaultdict(Counter)
    vocab = Counter()
    n = 0
    for eid, name, addr, country in _iter_s1(data_dir):
        n += 1
        c = (country or "").strip().lower() or "unknown"
        nm = name.lower() if name.isascii() else to_ascii(translit.text(name)[0]).lower()
        vocab.update(t for t in re.sub(r"[^a-z0-9 ]+", " ", nm.replace("'", "")).split())
        raw_parts = [p.strip() for p in addr.split(",")]
        raw_parts = [p for p in raw_parts if p]
        low = addr.lower()
        fr = c in ("france", "fr") or any(w in low for w in (" rue ", "rue ", " allee", " allée", "impasse", "chemin"))
        # street parts (with digits) are unique and useless for the gazetteer: skip before normalizing
        normed = [(None if _NUM.search(p) else norm(p, fr)) for p in raw_parts]
        normed = [p for p in normed if p != ""]
        for i, p in enumerate(normed):
            if p is None or len(p.split()) > 4 or _NOT_PLACE.intersection(p.split()):
                continue
            cnt[c][p] += 1
            if i == len(normed) - 1:
                last[c][p] += 1
    log(f"[gazetteer] scanned {n:,} S1 records")

    gaz = {}
    for c, counter in cnt.items():
        known = build_state_lookup(c)
        # countries with a state table (US, India) keep it; others (e.g. France) get mined regions
        states = [] if known else [p for p, k in counter.items()
                  if p not in known and k >= MIN_STATE and last[c][p] / k >= LAST_SHARE
                  and p.split()[-1] not in STREET_TYPES]
        stateset = set(states) | set(known)
        cities = {p: k for p, k in counter.items()
                  if k >= MIN_CITY and p not in stateset and p.split()[-1] not in STREET_TYPES
                  and p.split()[0] not in STREET_TYPES}
        gaz[c] = {"states": sorted(states), "cities": cities}
        log(f"[gazetteer] {c:>10}: {len(cities):,} cities, {len(states)} mined states/regions "
            f"(e.g. {states[:6]})")
    vocab = {w: k for w, k in vocab.most_common(300_000) if k >= 3}
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"countries": gaz, "vocab": vocab}, ensure_ascii=False))
    os.replace(tmp, out_path)  # atomic: never leaves a half-written file
    return {"countries": gaz, "vocab": vocab}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
