"""Low-level text helpers shared by name and address normalization."""
import re
import unicodedata

from anyascii import anyascii

from lexicon import HOMOGLYPHS

_WS = re.compile(r"\s+")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")


def is_nonlatin_char(c: str) -> bool:
    """True for letters/marks of non-Latin scripts (Devanagari, Tamil, ...)."""
    o = ord(c)
    if o < 0x250 or 0x1E00 <= o <= 0x1EFF:  # Basic/extended Latin
        return False
    cat = unicodedata.category(c)
    return cat[0] in ("L", "M")


def has_nonlatin(s: str) -> bool:
    return any(is_nonlatin_char(c) for c in s)


def clean_native_token(tok: str) -> str:
    """Strip punctuation from a native-script token but keep letters AND combining marks."""
    return "".join(c for c in tok if unicodedata.category(c)[0] in ("L", "M", "N"))


def to_ascii(s: str) -> str:
    """Accents and any leftover script -> ASCII (ISC-licensed anyascii)."""
    return anyascii(s)


def squash_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def homoglyph_candidate(tok: str) -> str:
    """Map every look-alike digit/symbol to its letter, no conditions ('8ig' -> 'big')."""
    return "".join(HOMOGLYPHS.get(c, c) for c in tok)


def fix_homoglyphs(tok: str) -> str:
    """G1obal -> global, C0FFEE -> coffee. Leaves numbers and ordinals alone."""
    if not tok or tok.isdigit() or tok.isalpha() or _ORDINAL.match(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if letters >= 3 and letters >= 3 * digits:
        return "".join(HOMOGLYPHS.get(c, c) for c in tok)
    return tok


def phonetic_squeeze(tok: str) -> str:
    """Cheap transliteration-robust key: drop vowels after the first char, collapse repeats.
    raam -> rm, ram -> rm, limtid -> lmtd, limited -> lmtd."""
    if not tok:
        return tok
    t = tok[0] + re.sub(r"[aeiouyhw]", "", tok[1:])
    return re.sub(r"(.)\1+", r"\1", t)


def levenshtein(a: str, b: str, max_d: int = 99) -> int:
    """Plain edit distance with early exit (small strings only)."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]
