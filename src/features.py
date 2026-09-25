"""
Root proxy for features.py
"""

import sys
from pathlib import Path

ber_src = Path(__file__).resolve().parent.parent / "business_entity_resolution" / "src"
if ber_src.exists():
    sys.path.insert(0, str(ber_src))

from features import main

if __name__ == "__main__":
    main()
