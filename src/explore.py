"""
explore.py - Exploratory Data Analysis for Business Entity Resolution

This file proxies or executes business_entity_resolution/src/explore.py
to allow execution directly from the workspace root (e.g. `python src/explore.py`).
"""

import sys
from pathlib import Path

# Add business_entity_resolution to sys.path
ber_src = Path(__file__).resolve().parent.parent / "business_entity_resolution" / "src"
if ber_src.exists():
    sys.path.insert(0, str(ber_src))

from explore import main

if __name__ == "__main__":
    main()
