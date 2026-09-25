"""
Root proxy for preprocess.py
"""

import sys
from pathlib import Path

ber_src = Path(__file__).resolve().parent.parent / "business_entity_resolution" / "src"
if ber_src.exists():
    sys.path.insert(0, str(ber_src))

from preprocess import (
    clean_text,
    expand_abbreviations,
    strip_legal_suffixes,
    extract_address_signals,
    main,
)

if __name__ == "__main__":
    main()
