"""Static lexicons used by Stage 0 normalization.

Only linguistic knowledge (abbreviations, legal-form spellings, state codes) lives
here - no business/entity data. Everything entity-specific is learned from the
provided training data (see translit_dict.py and gazetteer.py).
"""

# ---------------------------------------------------------------- names ----
# Canonical legal-form tag for every spelling we normalise to.
LEGAL_FORMS = {
    # US / generic English
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co", "companies": "co",
    "llc": "llc", "lc": "llc",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "lp": "lp", "llp": "llp", "lllp": "llp", "plc": "plc",
    "pc": "pc", "pllc": "pllc", "pa": "pa",
    # India
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt",
    "opc": "opc", "huf": "huf",
    # France / Europe
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "scs": "scs", "sca": "sca", "scop": "scop", "ei": "ei",
    "cie": "co", "compagnie": "co", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
# Multi-token spellings, collapsed before single-token lookup (after punctuation removal).
LEGAL_PHRASES = [
    ("l l c", "llc"), ("l l p", "llp"), ("p l c", "plc"), ("p c", "pc"),
    ("s a r l", "sarl"), ("s a s u", "sasu"), ("s a s", "sas"), ("e u r l", "eurl"),
    ("s c i", "sci"), ("s a", "sa"), ("pvt ltd", "pvt ltd"), ("p ltd", "pvt ltd"),
    ("pra li", "pvt ltd"), ("pra ltd", "pvt ltd"),
]
# Weak transliteration spellings of legal words (fallback when the learned dict misses).
TRANSLIT_LEGAL = {
    "praivet": "private", "prayvet": "private", "pravet": "private", "praiveta": "private",
    "limtid": "limited", "limitid": "limited", "limited": "limited", "limitad": "limited",
    "limitedd": "limited", "kampani": "company", "kampni": "company", "karporeshan": "corporation",
    "pra": "pvt", "li": "ltd", "eljepi": "llp", "elelpi": "llp",
}

# Leading honorifics / fillers that are noise in names.
NAME_PREFIX_NOISE = {"the", "mr", "mrs", "ms", "m s", "messrs", "m/s", "ms."}

# Descriptor words that are often added/dropped between sources. They are kept in
# name_core (they can be discriminative) but removed from name_key (blocking key).
GENERIC_WORDS = {
    "group", "groupe", "services", "service", "holdings", "holding", "international",
    "intl", "global", "enterprises", "enterprise", "solutions", "associates", "center",
    "centre", "and", "of", "et", "de", "du", "des", "la", "le", "les", "india", "usa",
    "france", "worldwide", "industries", "ventures",
    # honorifics injected as noise ("Smt ROCKMAN TRADERS", "Dr Moms Brothers")
    "dr", "smt", "shri", "sri", "shree", "mr", "mrs", "ms",
}
# Long legal words that arrive with typos ("Prfivete", "Limted"): first letter -> (word, tag)
FUZZY_LEGAL = {"p": ("private", "pvt"), "l": ("limited", "ltd"), "c": ("company", "co"),
               "i": ("incorporated", "inc")}

# Alias separators: "X formerly known as Y", "X dba Y" ...
ALIAS_SEPARATORS = [
    "formerly known as", "formerly", "f/k/a", "fka", "d/b/a", "dba", "doing business as",
    "a/k/a", "aka", "also known as", "t/a", "trading as",
]

# Homoglyph digits -> letters (applied only inside mostly-alphabetic name tokens).
HOMOGLYPHS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}

# Top-level domains stripped from domain-style names (energyvrtextile.com).
TLDS = {"com", "net", "org", "in", "co", "fr", "biz", "info", "us", "io", "co.in", "org.in",
        "net.in", "gov", "edu", "eu", "tech", "online", "store", "shop"}

# -------------------------------------------------------------- addresses ---
STREET_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "blvd": "boulevard", "bd": "boulevard",
    "ct": "court", "crt": "court", "pl": "place", "plz": "plaza", "hwy": "highway", "hiway": "highway",
    "pkwy": "parkway", "pky": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
    "terr": "terrace", "sq": "square", "mt": "mount", "ft": "fort", "expy": "expressway",
    "fwy": "freeway", "tpke": "turnpike", "cres": "crescent", "aly": "alley", "xing": "crossing",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "rte": "route", "rt": "route",
    "bldg": "building", "apts": "apartments", "apt": "apartment", "fl": "floor", "flr": "floor",
    # India
    "opp": "opposite", "opps": "opposite", "nr": "near", "bhd": "behind", "ngr": "nagar",
    "clny": "colony", "col": "colony", "sec": "sector", "sect": "sector", "ph": "phase",
    "extn": "extension", "ext": "extension", "tq": "taluk", "tal": "taluk", "dist": "district",
    "distt": "district", "vill": "village", "po": "postoffice", "ps": "policestation",
    "mkt": "market", "cplx": "complex", "indl": "industrial", "ind": "industrial",
    "estt": "estate", "chs": "society", "soc": "society", "stn": "station", "gali": "gali",
}
# Applied instead of STREET_ABBR entries when the address reads as French.
STREET_ABBR_FR = {
    "st": "saint", "ste": "sainte", "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard",
    "bld": "boulevard", "bvd": "boulevard", "all": "allee", "imp": "impasse", "ch": "chemin",
    "chem": "chemin", "rte": "route", "pl": "place", "fg": "faubourg", "fbg": "faubourg",
    "sq": "square", "crs": "cours", "qu": "quai", "res": "residence", "lot": "lotissement",
    "zi": "zone industrielle", "za": "zone artisanale", "pdt": "president", "gal": "general",
    "mal": "marechal", "cdt": "commandant", "dr": "docteur",
}
FRENCH_HINTS = {"rue", "allee", "impasse", "chemin", "boulevard", "avenue", "place", "quai",
                "cours", "faubourg", "du", "des", "de", "la", "le", "les", "lieu", "dit", "cedex"}

UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "room", "rm", "flat", "floor", "fl",
              "flr", "office", "shop", "bay", "box", "lot", "space", "spc", "trlr", "bldg", "building"}
HOUSE_PREFIXES = {"no", "nos", "number", "h", "hno", "house", "door", "dno", "plot", "kh",
                  "khasra", "survey", "sy", "serve", "s", "c", "d", "g", "b", "a", "f", "l", "m"}
LANDMARK_WORDS = {"near", "opposite", "behind", "beside", "adjacent", "next", "facing",
                  "above", "below", "opp", "nr", "bhd", "besides"}
NULL_TOKENS = {"<null>", "null", "none", "nan", "n/a", "na", "unknown", "-"}

# State / province names. Canonical value = key used in the `state` column.
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}
INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ch": "chandigarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand", "jk": "jammu and kashmir",
    "ka": "karnataka", "kl": "kerala", "la": "ladakh", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha",
    "or": "odisha", "orissa": "odisha", "pb": "punjab", "py": "puducherry", "pondicherry": "puducherry",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana", "tg": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "uttaranchal": "uttarakhand", "wb": "west bengal", "dl": "delhi", "nct of delhi": "delhi",
    "new delhi": "delhi", "an": "andaman and nicobar islands", "dn": "dadra and nagar haveli",
    "dd": "daman and diu", "ld": "lakshadweep", "tamilnadu": "tamil nadu",
}
# Country label -> state table. Unknown labels (e.g. France) use states mined from data.
STATE_TABLES = {"us": US_STATES, "usa": US_STATES, "united states": US_STATES,
                "india": INDIA_STATES, "in": INDIA_STATES}

# Ordinal words in street names -> numeric form ("Fourteenth Ave" == "14th Ave"; French "premier" == "1er")
ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
    "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th",
    "twentieth": "20th", "1er": "1st", "1ere": "1st", "premier": "1st", "premiere": "1st",
}
# India: "C/O Mohit Maik", "S/O Ram" name a person, not a place -> kept out of street tokens
CARE_OF = ("c o ", "s o ", "w o ", "d o ", "care of ")
