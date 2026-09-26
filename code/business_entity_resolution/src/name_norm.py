"""Business-name normalization (Stage 0).

Produces, per record:
  name_clean  full normalized name, legal words mapped to canonical tags, order kept
  name_core   name without legal-form words (what the business is called)
  name_key    sorted, de-duplicated core tokens minus generic descriptors (blocking key)
  name_alt    cores of alias parts ("X formerly known as Y", "X dba Y"), '|' separated
  legal       sorted canonical legal tags, e.g. "llc" or "ltd pvt"
  name_translit 0 = Latin input, 1 = anyascii fallback used, 2 = fully from learned dictionary
  name_is_domain 1 if the name was a web domain (energyvrtextile.com)
"""
import math
import re
import unicodedata

from lexicon import (ALIAS_SEPARATORS, FUZZY_LEGAL, GENERIC_WORDS, LEGAL_FORMS, LEGAL_PHRASES,
                     TLDS, TRANSLIT_LEGAL)
from text_utils import fix_homoglyphs, homoglyph_candidate, levenshtein, squash_ws, to_ascii

_ID_NOISE = [
    re.compile(r"[\(\[]\s*(?:id|ref|no|#)\s*[:#.]?\s*\d+\s*[\)\]]"),
    re.compile(r"\b(?:id|ref)\s*[:#]\s*\d+"),
    re.compile(r"#\s*\d{2,}"),
    re.compile(r"-\s*\d{3}\s?\d{3}\b"),          # "gurgaon-122001" PIN glued to a word
]
_ALIAS = re.compile(r"\s(?:" + "|".join(re.escape(a) for a in sorted(ALIAS_SEPARATORS, key=len, reverse=True)) + r")\s")
_ACRONYM_DOTS = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)((?:\.[a-z]{2,6}){1,2})/?$")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_LEGAL_PHRASES = [(re.compile(r"\b" + a + r"\b"), b) for a, b in LEGAL_PHRASES]


class Segmenter:
    """Split glued words (domains) using S1 token frequencies: 'energyvrtextile' -> 'energy vr textile'."""

    def __init__(self, vocab: dict, max_len: int = 20):
        total = float(sum(vocab.values())) or 1.0
        self.cost = {w: -math.log(c / total) for w, c in vocab.items() if len(w) > 1 or w in ("a",)}
        self.unk = -math.log(1.0 / total) + 5.0
        self.max_len = max_len

    def split(self, s: str):
        n = len(s)
        best = [0.0] + [float("inf")] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                w = s[j:i]
                c = self.cost.get(w, self.unk * len(w))
                if best[j] + c < best[i]:
                    best[i], back[i] = best[j] + c, j
        out, i = [], n
        while i > 0:
            out.append(s[back[i]:i]); i = back[i]
        return out[::-1]


def _basic_clean(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("+", " and ")
    for rx in _ID_NOISE:
        s = rx.sub(" ", s)
    s = re.sub(r"['’`´]", "", s)                         # kayley's -> kayleys
    s = _ACRONYM_DOTS.sub(lambda m: m.group(0).replace(".", ""), s)  # l.l.c. -> llc
    s = _PUNCT.sub(" ", s)
    return squash_ws(s)


def _tokens(s: str, from_native: bool):
    s = s
    for rx, rep in _LEGAL_PHRASES:
        s = rx.sub(rep, s)
    toks = [fix_homoglyphs(t) for t in s.split()]
    if from_native:
        toks = [TRANSLIT_LEGAL.get(t, t) for t in toks]
    # leading honorific / filler noise
    while toks and toks[0] in ("the", "mr", "mrs", "ms", "messrs", "mx"):
        toks = toks[1:]
    if len(toks) > 1 and toks[0] == "m" and toks[1] == "s":
        toks = toks[2:]
    # collapse immediate repeats (L.L.C. L.L.C.)
    dedup = []
    for t in toks:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return dedup


def _fuzzy_legal(toks, vocab_cost):
    """Repair typo'd legal words near the end of a name.
    Rule 1 (context): a p-word right before ltd/limited is 'private' (Plrvae/Prislie Limited).
    Rule 2 (distance): last-3 tokens within edit distance of private/limited/company/incorporated."""
    out = list(toks)
    n = len(out)
    for i in range(max(1, n - 3), n):  # never rewrite the first token
        t = out[i]
        if t in LEGAL_FORMS or len(t) < 4 or not t[0] in FUZZY_LEGAL:
            continue
        if vocab_cost is not None and t in vocab_cost and vocab_cost[t] < 12:  # common real word: keep
            continue
        word, tag = FUZZY_LEGAL[t[0]]
        nxt = out[i + 1] if i + 1 < n else ""
        if t[0] == "p" and LEGAL_FORMS.get(nxt) == "ltd" and 4 <= len(t) <= 10:
            out[i] = tag
        elif levenshtein(t, word, 3) <= (2 if len(word) <= 7 else 3):
            out[i] = tag
    return out


def _split_legal(toks):
    legal, core = [], []
    for t in toks:
        if t in LEGAL_FORMS:
            legal.append(LEGAL_FORMS[t])
        else:
            core.append(t)
    if not core:  # name made only of legal words: keep them as the core
        core = [t for t in toks]
    return core, legal


class NameNormalizer:
    def __init__(self, translit, segmenter: Segmenter = None):
        self.tr = translit
        self.seg = segmenter

    def _post(self, toks):
        """Segment glued tokens (#hitechhospitality, promotersprivate), then repair legal typos."""
        cost = self.seg.cost if self.seg is not None else None
        if cost is not None:
            # vocabulary-confirmed homoglyph repair for short tokens the heuristic skips ('8ig' -> 'big')
            fixed = []
            for t in toks:
                if (len(t) >= 3 and sum(c.isalpha() for c in t) >= 2
                        and not t.isalpha() and not t.isdigit()):
                    cand = homoglyph_candidate(t)
                    if cand.isalpha() and cand in cost and cost[cand] < cost.get(t, float("inf")) - 2:
                        t = cand
                fixed.append(t)
            toks = fixed
            out = []
            for t in toks:
                if len(t) >= 10 and not t.isdigit():
                    parts = self.seg.split(t)
                    # split only when the pieces are jointly more likely than the glued token
                    if (len(parts) > 1 and all(p in cost for p in parts)
                            and sum(cost[p] for p in parts) < cost.get(t, float("inf"))):
                        out.extend(parts)
                        continue
                out.append(t)
            toks = [LEGAL_FORMS.get(t, t) if t in ("private", "limited") else t for t in out]
        return _fuzzy_legal(toks, cost)

    def normalize(self, raw: str) -> dict:
        raw = unicodedata.normalize("NFKC", raw or "").strip()
        out = dict(name_clean="", name_core="", name_key="", name_alt="", legal="",
                   name_translit=0, name_is_domain=0)
        if not raw:
            return out
        s, n_nat, n_learn = self.tr.text(raw)
        if n_nat:
            out["name_translit"] = 2 if n_learn == n_nat else 1
        s = to_ascii(s).lower().strip()

        # web-domain names
        m = _DOMAIN.match(s.replace(" ", ""))
        if m and m.group(2).lstrip(".") in TLDS | {t.split(".")[-1] for t in TLDS} and "." in s:
            out["name_is_domain"] = 1
            body = m.group(1).replace("-", " ")
            if self.seg is not None:
                body = " ".join(" ".join(self.seg.split(w)) for w in body.split())
            s = body

        # aliases: "x formerly known as y", "x dba y"
        pieces = [p for p in _ALIAS.split(" " + s + " ") if p.strip()]
        cleaned = [self._post(_tokens(_basic_clean(p), n_nat > 0)) for p in pieces]
        cleaned = [c for c in cleaned if c] or [[]]
        main = cleaned[-1]                  # observed: the real/S1 name tends to be the last part
        alts = cleaned[:-1]

        core, legal = _split_legal(main)
        full = [LEGAL_FORMS.get(t, t) for t in main]
        key = sorted(set(t for t in core if t not in GENERIC_WORDS)) or sorted(set(core))
        alt_cores = []
        for a in alts:
            c, lg = _split_legal(a)
            legal += lg
            alt_cores.append(" ".join(c))
        out.update(
            name_clean=" ".join(full),
            name_core=" ".join(core),
            name_key=" ".join(key),
            name_alt=" | ".join(x for x in alt_cores if x),
            legal=" ".join(sorted(set(legal))),
        )
        return out
