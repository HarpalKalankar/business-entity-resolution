"""Address normalization + light parsing (Stage 0).

Parts come in any order ("IA, Iowa City, 1064 Newton Rd"), so every comma part is
*classified* (postcode / state / city / street / landmark / locality) instead of
being read by position. Output fields are plain strings so they store well in Parquet.
"""
import re
import unicodedata

from lexicon import (CARE_OF, ORDINAL_WORDS, LANDMARK_WORDS, NULL_TOKENS, STATE_TABLES,
                     STREET_ABBR, STREET_ABBR_FR, HOUSE_PREFIXES)
from text_utils import has_nonlatin, squash_ws, to_ascii

STREET_TYPES = {"road", "street", "avenue", "drive", "lane", "boulevard", "court", "place",
                "highway", "parkway", "circle", "trail", "terrace", "way", "square", "plaza",
                "route", "rue", "allee", "impasse", "chemin", "quai", "cours", "marg", "path",
                "expressway", "freeway", "turnpike", "crescent", "alley", "pike", "loop", "run",
                "row", "walk", "bypass", "street no"}

_UNIT = re.compile(
    r"\b(?:unit|apt|apartment|suite|ste|room|rm|flat|office|shop|space|spc|trlr|lot|box|bay)\b"
    r"[\s#:.\-]*(?:(?:unit|apt|apartment|suite|ste|room|rm|flat|no)\b[\s#:.\-]*)?"
    r"([a-z]?\d+[a-z]?|[a-z]\b)")
_FLOOR = re.compile(r"\b(\d{1,3})\s*(?:st|nd|rd|th)?\s*(?:floor|flr|fl)\b|\b(?:floor|flr)\s*(?:no\s*)?(\d{1,3})\b")
_NAMED_FLOOR = re.compile(r"\b(ground|first|second|third|fourth|fifth|basement|upper|lower|mezzanine|top)\s+(?:floor|flr|fl)\b|\brez\s+de\s+chaussee\b|\brdc\b")
_PMB = re.compile(r"\b(?:pmb|p\s*o\s*box|po\s*box|post\s*box)\s*#?\s*(\d+)")
_PIN_INLINE = re.compile(r"(?:\bpin(?:code)?\b|\bzip\b)?\s*[:\-]\s*(\d{3}\s?\d{3})\b|\bpin(?:code)?\s*(\d{3}\s?\d{3})\b")
_STATE_ZIP = re.compile(r"^([a-z]{2})\s+(\d{5})(?:\s*-\s*\d{4})?$")
_ZIP_CITY = re.compile(r"^(\d{5})\s+([a-z].*)$")
_NUM = re.compile(r"\d+")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


_FR_STRONG = {"rue", "allee", "impasse", "chemin", "quai", "faubourg", "cedex", "lieu"}


def is_french(tokens, country: str) -> bool:
    """Language cue, not a hard-coded country list: French street words switch the abbreviation map."""
    return country in ("france", "fr") or bool(_FR_STRONG.intersection(tokens))


def expand_tokens(tokens, french: bool):
    out = []
    for t in tokens:
        if t in ORDINAL_WORDS:
            out.append(ORDINAL_WORDS[t])
        elif french and t in STREET_ABBR_FR:
            out.append(STREET_ABBR_FR[t])
        else:
            out.append(STREET_ABBR.get(t, t))
    return out


def norm_part_text(p: str, french: bool) -> str:
    """ASCII, lower, punctuation -> space, abbreviations expanded."""
    p = to_ascii(p).lower()
    p = p.replace("'", " ").replace("’", " ")
    p = _PUNCT.sub(" ", p)
    toks = [t for t in p.split() if t not in NULL_TOKENS]
    # French elision glued by noise: lamiral -> l amiral handled loosely by leaving as is
    return " ".join(expand_tokens(toks, french))


def build_state_lookup(country: str, mined_states=()):
    table = STATE_TABLES.get(country, {})
    lk = {}
    for code, name in table.items():
        lk[code] = name
        lk[name] = name
    for s in mined_states:
        lk.setdefault(s, s)
    return lk


class AddressNormalizer:
    def __init__(self, translit, gazetteer):
        self.tr = translit
        self.gaz = gazetteer          # dict country -> {"states": [...], "cities": {name: count}}
        self._state_lk = {}

    def _states(self, country):
        if country not in self._state_lk:
            mined = self.gaz.get(country, {}).get("states", [])
            self._state_lk[country] = build_state_lookup(country, mined)
        return self._state_lk[country]

    def normalize(self, raw: str, country: str) -> dict:
        country = (country or "").strip().lower() or "unknown"
        out = dict(addr_clean="", house_num="", all_nums="", street="", locality="", city="",
                   state="", postcode="", unit="", landmark="", addr_missing=1, addr_translit=0)
        raw = unicodedata.normalize("NFKC", raw or "").strip()
        if not raw or raw.lower() in NULL_TOKENS:
            return out
        out["addr_missing"] = 0
        states = self._states(country)
        cities = self.gaz.get(country, {}).get("cities", {})

        # ---- per-part transliteration (learned whole-part table, then tokens)
        parts = []
        for p in raw.split(","):
            if has_nonlatin(p):
                out["addr_translit"] = 1
                p = self.tr.part(p)
            p = to_ascii(p).lower().strip()
            if p and p not in NULL_TOKENS:
                parts.append(p)
        all_tokens = set(_PUNCT.sub(" ", " ".join(parts)).split())
        french = is_french(all_tokens, country)

        units, postcode, landmark = [], "", []
        clean_parts = []
        for p in parts:
            p = re.sub(r"<\s*null\s*>", " ", p)
            # PMB / PO box -> unit
            for m in _PMB.finditer(p):
                units.append("box" + m.group(1))
            p = _PMB.sub(" ", p)
            # inline PIN "tirunelveli-627 002"
            m = _PIN_INLINE.search(p)
            if m:
                postcode = postcode or (m.group(1) or m.group(2)).replace(" ", "")
                p = _PIN_INLINE.sub(" ", p)
            # floors and units
            for m in _FLOOR.finditer(p):
                units.append("fl" + (m.group(1) or m.group(2)))
            p = _FLOOR.sub(" ", p)
            for m in _NAMED_FLOOR.finditer(p):
                units.append("fl" + (m.group(1) or "ground")[:3])
            p = _NAMED_FLOOR.sub(" ", p)
            if not french or "ste" not in p.split():
                for m in _UNIT.finditer(p):
                    units.append(m.group(1))
                p = _UNIT.sub(" ", p)
            p = squash_ws(p)
            if p:
                clean_parts.append(p)

        street_parts, loc_parts, city_cands, state_found = [], [], [], []
        n_numeric_parts = sum(bool(_NUM.search(p)) for p in clean_parts)
        for p in clean_parts:
            raw_p = p
            # "tx 75001"  /  "33000 bordeaux"
            m = _STATE_ZIP.match(p)
            if m and m.group(1) in states:
                state_found.append(states[m.group(1)]); postcode = postcode or m.group(2); continue
            m = _ZIP_CITY.match(p)
            if m and french:
                postcode = postcode or m.group(1); p = m.group(2)
            compact = p.replace(" ", "")
            if compact.isdigit() and len(compact) in (5, 6) and n_numeric_parts > 1:
                postcode = postcode or compact; continue
            t = norm_part_text(p, french)
            if not t:
                continue
            if t in states:
                state_found.append(states[t]); continue
            if (t + " ").startswith(CARE_OF):   # "c/o mohit maik": a person, not a location
                landmark.append(t)
                continue
            toks = t.split()
            # landmark: split at first landmark word
            lm_idx = next((i for i, w in enumerate(toks) if w in LANDMARK_WORDS
                           or (w == "next" and i + 1 < len(toks) and toks[i + 1] == "to")
                           or (w in ("c", "s", "w", "d") and i + 1 < len(toks) and toks[i + 1] == "o"
                               and i + 2 < len(toks))), None)
            if lm_idx is not None:
                landmark.append(" ".join(toks[lm_idx:]))
                toks = toks[:lm_idx]
                if not toks:
                    continue
                t = " ".join(toks)
            if t in cities:
                city_cands.append(t); continue
            if _NUM.search(t) or toks[-1] in STREET_TYPES or toks[0] in STREET_TYPES:
                street_parts.append(t)
            else:
                loc_parts.append(t)

        # house number = first number of the first numeric street part
        # numbers, ignoring ordinals used as street names ("32nd avenue", "8th street")
        nums, house = [], ""
        for sp in street_parts:
            for w in sp.split():
                if _ORDINAL.fullmatch(w):
                    continue
                for d in _NUM.findall(w):
                    nums.append(d)
                    if not house:
                        house = str(int(d)) if len(d) < 10 else d
        street_words = []
        for sp in street_parts:
            for w in sp.split():
                if _NUM.fullmatch(w) or w in HOUSE_PREFIXES or re.fullmatch(r"\d+[a-z]{1,3}", w) and not re.fullmatch(r"\d+(st|nd|rd|th)", w):
                    continue
                street_words.append(w)

        # city = most frequent gazetteer hit; others become locality
        city = ""
        if city_cands:
            city = max(city_cands, key=lambda c: cities.get(c, 0))
            loc_parts += [c for c in city_cands if c != city]
        out.update(
            house_num=house,
            all_nums=" ".join(sorted({str(int(n)) if len(n) < 10 else n for n in nums})),
            street=" ".join(street_words),
            locality=" | ".join(dict.fromkeys(loc_parts)),
            city=city,
            state=" | ".join(dict.fromkeys(state_found)),
            postcode=postcode,
            unit=" ".join(dict.fromkeys(units)),
            landmark=" | ".join(landmark),
        )
        out["addr_clean"] = squash_ws(" ".join(
            street_parts + list(dict.fromkeys(loc_parts)) + [city] + list(dict.fromkeys(state_found))))
        return out
