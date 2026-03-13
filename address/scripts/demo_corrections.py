#!/usr/bin/env python3
"""Demo: correct OCR-corrupted addresses using the full pipeline.

This script requires a running PostgreSQL instance with the index already
built (via build_index.py).  It runs the 5 built-in test cases and prints
the results.

Usage:
    uv run python scripts/demo_corrections.py
    uv run python scripts/demo_corrections.py --conninfo "dbname=address_correction"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from address_correction import db
from address_correction.matcher import AddressMatcher
from address_correction.sample_data import TEST_CASES
from address_correction.vectorizer import AddressVectorizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo address corrections")
    parser.add_argument(
        "--conninfo",
        default="dbname=address_correction",
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory containing the fitted vectorizer",
    )
    args = parser.parse_args()

    vectorizer = AddressVectorizer.load(Path(args.model_dir))
    conn = db.get_connection(args.conninfo)
    matcher = AddressMatcher(vectorizer, conn)

    print("=" * 72)
    print("Address Correction Demo — 5 Retail Locations with OCR Errors")
    print("=" * 72)

    passed = 0
    for case in TEST_CASES:
        result = matcher.correct(case["vendor_name"], case["ocr_address"])

        ok = result.matched and result.corrected_address == case["expected_address"]
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"\n[{status}] {case['description']}")
        print(f"  Vendor:    {case['vendor_name']}")
        print(f"  OCR input: {case['ocr_address']}")
        if result.matched:
            print(f"  Corrected: {result.corrected_address}")
            print(f"  City:      {result.corrected_city}")
            print(f"  Conf:      {result.confidence:.3f}")
        else:
            print("  Corrected: (no match)")
        print(f"  Expected:  {case['expected_address']}")

    print(f"\n{'=' * 72}")
    print(f"Results: {passed}/{len(TEST_CASES)} passed")
    conn.close()


if __name__ == "__main__":
    main()
