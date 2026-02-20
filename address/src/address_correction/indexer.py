"""Build the pgvector index from reference data.

Fits the TF-IDF vectorizer on the reference records, computes embeddings,
and loads everything into PostgreSQL.
"""

from pathlib import Path

import psycopg

from . import db
from .download import load_parquet
from .sample_data import REFERENCE_DATA
from .vectorizer import AddressVectorizer


def build_index(
    conn: psycopg.Connection,
    vectorizer: AddressVectorizer,
    records: list[dict],
) -> AddressVectorizer:
    """Fit the vectorizer on *records*, create the table, and insert rows."""
    vendor_names = [r["vendor_name"] for r in records]
    addresses = [r["address"] for r in records]

    vectorizer.fit(vendor_names, addresses)
    embeddings = vectorizer.transform(vendor_names, addresses)

    db.drop_tables(conn)
    db.create_tables(conn, dimensions=vectorizer.dimensions)
    db.insert_references(conn, records, embeddings)

    return vectorizer


def build_index_from_parquet(
    conn: psycopg.Connection,
    vectorizer: AddressVectorizer,
    parquet_path: str | Path,
) -> AddressVectorizer:
    """Load a Parquet file (from download_overture_places) and build the index."""
    records = load_parquet(parquet_path)
    return build_index(conn, vectorizer, records)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the address correction index")
    parser.add_argument(
        "--conninfo",
        default="dbname=address_correction",
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Path to Overture Parquet file. If omitted, uses built-in sample data.",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory to save the fitted vectorizer (default: model/)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=256,
        help="Target embedding dimensions (default: 256)",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.conninfo)
    vectorizer = AddressVectorizer(n_components=args.dimensions)

    if args.parquet:
        print(f"Building index from {args.parquet}...")
        build_index_from_parquet(conn, vectorizer, args.parquet)
    else:
        print("Building index from built-in sample data...")
        build_index(conn, vectorizer, REFERENCE_DATA)

    model_dir = Path(args.model_dir)
    vectorizer.save(model_dir)
    print(f"Vectorizer saved to {model_dir}/")
    print("Done.")
    conn.close()


if __name__ == "__main__":
    main()
