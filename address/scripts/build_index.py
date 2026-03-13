#!/usr/bin/env python3
"""Build the pgvector index from Overture data or built-in sample data.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --parquet data/overture_places.parquet
    uv run python scripts/build_index.py --conninfo "dbname=address_correction"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from address_correction.indexer import main

if __name__ == "__main__":
    main()
