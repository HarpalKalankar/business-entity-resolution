"""Learn a native-script -> Latin dictionary from the TRAINING ground truth.

Idea: many Source 2/3 records spell the name (or state) in an Indic script while
their Source 1 twin is in Latin script. Aligning them gives a data-driven
transliteration + translation table, e.g.
    'ਪਾਇਨੀਅਰ' -> 'pioneer', 'ਪ੍ਰਾਈਵੇਟ' -> 'private', 'தமிழ்நாடு' -> 'tamil nadu'.

Two tables are learned:
  * token table  (name tokens, aligned by position when token counts match)
  * part table   (whole comma-separated address parts, by co-occurrence / PMI)
Unseen tokens fall back to anyascii + phonetic squeeze in the normalizers.
Uses only the provided training data.
"""
import json
import os
import math
import re
import sys
from collections import Counter, defaultdict

from text_utils import clean_native_token, has_nonlatin, to_ascii, squash_ws

_LAT_PUNCT = re.compile(r"[^a-z0-9 ]+")
# fast pre-filter: any character outside Latin ranges and common punctuation blocks
_MAYBE_NATIVE = re.compile(r"[^\x00-\u024F\u1E00-\u1EFF\u2000-\u206F\u20A0-\u20CF]")


def _latin_tokens(s: str):
    s = to_ascii(s).lower()
    s = re.sub(r"[\"'`’.]", "", s)
    s = _LAT_PUNCT.sub(" ", s)
    return [t for t in s.split() if t]


def _native_tokens(s: str):
    return [clean_native_token(t) for t in s.split() if clean_native_token(t)]


def _latin_part(p: str) -> str:
    return " ".join(_latin_tokens(p))


def _native_part(p: str) -> str:
    return " ".join(_native_tokens(p))


def _read_tsv(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            yield parts


def build(data_dir: str, out_path: str, log=print):
    train = f"{data_dir}/train"
    # 1) native-script S2/S3 records
    native = {}
    for src in ("source2", "source3"):
        with open(f"{train}/train_{src}.tsv", encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                if not _MAYBE_NATIVE.search(line):
                    continue
                p = line.rstrip("\n").split("\t") + ["", "", ""]
                eid, name, addr = p[0], p[1], p[2]
                if has_nonlatin(name) or has_nonlatin(addr):
                    native[eid] = (name, addr)
    log(f"[translit] native-script S2/S3 records: {len(native):,}")

    # 2) owner S1 of each native record
    owner = {}
    with open(f"{train}/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            for x in ids.split(","):
                if x in native:
                    owner[x] = s1
    need = set(owner.values())
    s1 = {}
    for eid, name, addr, country in _read_tsv(f"{train}/train_source1.tsv"):
        if eid in need:
            s1[eid] = (name, addr)
    log(f"[translit] linked to {len(need):,} S1 entities")

    tok_pair = defaultdict(Counter)   # native token -> Counter(latin token)
    tok_cnt = Counter()
    part_pair = defaultdict(Counter)  # native part -> Counter(latin part)
    part_cnt = Counter()
    lat_part_cnt = Counter()

    for x, s in owner.items():
        n_name, n_addr = native[x]
        l_name, l_addr = s1[s]
        # ---- names: position alignment when token counts agree
        if has_nonlatin(n_name):
            nt, lt = _native_tokens(n_name), _latin_tokens(l_name)
            nt_nat = [t for t in nt if has_nonlatin(t)]
            if nt and len(nt) == len(lt):
                for a, b in zip(nt, lt):
                    if has_nonlatin(a):
                        tok_pair[a][b] += 1
            for a in set(nt_nat):
                tok_cnt[a] += 1
        # ---- address parts: co-occurrence of whole comma parts
        if has_nonlatin(n_addr):
            lparts = {_latin_part(p) for p in l_addr.split(",")} - {""}
            for lp in lparts:
                lat_part_cnt[lp] += 1
            for p in n_addr.split(","):
                if has_nonlatin(p):
                    npart = _native_part(p)
                    if not npart:
                        continue
                    part_cnt[npart] += 1
                    for lp in lparts:
                        part_pair[npart][lp] += 1

    tokens = {}
    for a, c in tok_pair.items():
        b, k = c.most_common(1)[0]
        if k / max(1, tok_cnt[a]) >= 0.4 and (k >= 2 or tok_cnt[a] == 1):
            tokens[a] = b
    parts = {}
    for a, c in part_pair.items():
        best, best_s = None, 0.0
        for b, k in c.items():
            if k < 2:
                continue
            s_ = k * k / (part_cnt[a] * lat_part_cnt[b])  # squared PMI-like score
            if s_ > best_s:
                best, best_s = b, s_
        if best is not None and best_s >= 0.05:
            parts[a] = best
    log(f"[translit] learned {len(tokens):,} name tokens, {len(parts):,} address parts")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"tokens": tokens, "parts": parts}, ensure_ascii=False))
    os.replace(tmp, out_path)  # atomic: never leaves a half-written file
    return {"tokens": tokens, "parts": parts}


class Transliterator:
    """Apply the learned tables; fall back to anyascii."""

    def __init__(self, table: dict):
        self.tokens = table.get("tokens", {})
        self.parts = table.get("parts", {})

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def token(self, tok: str):
        """Return (latin_text, learned: bool)."""
        key = clean_native_token(tok)
        if key in self.tokens:
            return self.tokens[key], True
        return to_ascii(tok), False

    def text(self, s: str):
        """Transliterate a name-like string token by token. Returns (text, n_native, n_learned)."""
        if not has_nonlatin(s):
            return s, 0, 0
        out, n_nat, n_learn = [], 0, 0
        for t in s.split():
            if has_nonlatin(t):
                n_nat += 1
                lat, learned = self.token(t)
                n_learn += learned
                out.append(lat)
            else:
                out.append(t)
        return " ".join(out), n_nat, n_learn

    def part(self, p: str):
        """Transliterate one comma part of an address."""
        if not has_nonlatin(p):
            return p
        key = _native_part(p)
        if key in self.parts:
            return self.parts[key]
        return self.text(p)[0]


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
