#!/usr/bin/env python3
"""Download Overture Maps Places data for target countries.

Usage:
    uv run python scripts/download_overture.py
    uv run python scripts/download_overture.py --countries US FR --output data/us_fr.parquet
"""

import sys
from pathlib import Path

# Allow running from the address/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from address_correction.download import main

if __name__ == "__main__":
    main()
