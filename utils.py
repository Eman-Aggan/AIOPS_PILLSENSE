"""Pure helper functions (no API calls), so they can be unit-tested in CI."""
import os, re, json, difflib

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "config", "brands.json"), encoding="utf-8") as f:
    BRANDS = json.load(f)

KNOWN = set(BRANDS.keys())

LOOKUP = {}
for generic, aliases in BRANDS.items():
    LOOKUP[generic] = generic
    for a in aliases:
        LOOKUP[a.lower()] = generic


def normalize_name(raw):
    """Converts any name (brand / Arabic / with dose / misspelled) to the generic name."""
    if not raw:
        return "unknown"
    name = raw.lower().strip()
    name = re.sub(r"\d+(\.\d+)?\s*(mg|g|mcg|ml|مجم|جم)", "", name)
    name = re.sub(r"\b(tablets?|capsules?|syrup|cream|film.coated)\b", "", name)
    name = re.sub(r"(اقراص|أقراص|كبسولات)", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    if name in LOOKUP:                                   # 1) exact match
        return LOOKUP[name]
    for alias, generic in LOOKUP.items():                # 2) contains a known alias
        if alias in name:
            return generic
    close = difflib.get_close_matches(name, LOOKUP.keys(), n=1, cutoff=0.8)  # 3) typo
    if close:
        return LOOKUP[close[0]]
    return name or "unknown"                             # 4) unknown: return as-is


def find_drugs_in_text(text):
    """Find every known drug (generic, brand or Arabic) mentioned in a sentence."""
    t = text.lower()
    found = []
    for alias, generic in LOOKUP.items():
        if alias.isascii():
            hit = re.search(rf"\b{re.escape(alias)}\b", t)
        else:
            hit = alias in t
        if hit and generic not in found:
            found.append(generic)
    return found
