import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import normalize_name, find_drugs_in_text, KNOWN


def test_brand_with_dose():
    assert normalize_name("Panadol 500mg") == "paracetamol"

def test_arabic():
    assert normalize_name("بنادول") == "paracetamol"

def test_typo():
    assert normalize_name("paracetmol") == "paracetamol"

def test_brand_with_words():
    assert normalize_name("Cataflam 50 mg tablets") == "diclofenac"

def test_unknown_drug_kept():
    assert normalize_name("insulin") == "insulin"
    assert "insulin" not in KNOWN

def test_empty():
    assert normalize_name("") == "unknown"

def test_find_two_drugs_english():
    assert set(find_drugs_in_text("Can I take brufen with warfarin?")) == {"ibuprofen", "warfarin"}

def test_find_two_drugs_arabic():
    assert set(find_drugs_in_text("ينفع آخد بنادول مع بروفين؟")) == {"paracetamol", "ibuprofen"}

def test_all_leaflets_have_brands():
    files = {f[:-4] for f in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))}
    assert files == KNOWN
