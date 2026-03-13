"""Download Overture Maps Places data for target countries.

Uses DuckDB to query GeoParquet files directly from S3 and write a local
Parquet file containing (vendor_name, address, city, postcode, country).
"""

from pathlib import Path

import duckdb

# Overture Maps release — update this when a newer release is available.
DEFAULT_RELEASE = "2024-12-18.0"
S3_BASE = "s3://overturemaps-us-west-2/release"
DEFAULT_COUNTRIES = ("US", "CA", "FR", "DE")


def download_overture_places(
    output_path: str | Path,
    *,
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    release: str = DEFAULT_RELEASE,
) -> Path:
    """Download Overture Maps Places for the given countries to a local Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    places_url = f"{S3_BASE}/{release}/theme=places/type=place/*"
    country_list = ", ".join(f"'{c}'" for c in countries)

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region = 'us-west-2';")

    query = f"""
        COPY (
            SELECT
                names.primary           AS vendor_name,
                addresses[1].freeform   AS address,
                addresses[1].locality   AS city,
                addresses[1].postcode   AS postcode,
                addresses[1].country    AS country
            FROM read_parquet('{places_url}', filename=true, hive_partitioning=1)
            WHERE addresses[1].country IN ({country_list})
              AND names.primary          IS NOT NULL
              AND addresses[1].freeform  IS NOT NULL
        ) TO '{output_path}' (FORMAT PARQUET)
    """

    con.execute(query)
    con.close()

    return output_path


def load_parquet(path: str | Path) -> list[dict]:
    """Read a downloaded Parquet file into a list of record dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return table.to_pylist()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Overture Maps Places data")
    parser.add_argument(
        "-o", "--output",
        default="data/overture_places.parquet",
        help="Output Parquet file path (default: data/overture_places.parquet)",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        help="ISO country codes to include (default: US CA FR DE)",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"Overture Maps release tag (default: {DEFAULT_RELEASE})",
    )
    args = parser.parse_args()

    print(f"Downloading Overture Maps Places for {args.countries} (release {args.release})...")
    out = download_overture_places(
        args.output,
        countries=tuple(args.countries),
        release=args.release,
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
