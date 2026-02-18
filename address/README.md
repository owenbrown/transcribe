# Address Error Correction with Vendor Name

Corrects OCR-corrupted addresses on receipts by matching against known
(vendor name, address) pairs from Overture Maps data.

**Approach:** Character n-gram TF-IDF embeddings stored in PostgreSQL via
pgvector.  A two-stage pipeline retrieves candidates by vector similarity
then reranks by string similarity on the vendor name and address components.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 14+ (already installed on the host)

## 1. Install PostgreSQL Extensions

The system uses two PostgreSQL extensions: **pgvector** for vector similarity
search and **pg_trgm** for trigram-based fuzzy text matching on vendor names.

### pgvector

```bash
# Ubuntu / Debian (match your PostgreSQL major version)
sudo apt install postgresql-16-pgvector

# macOS with Homebrew
brew install pgvector

# From source (any platform)
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
```

After installing, the extension is activated automatically by the setup
script (`CREATE EXTENSION IF NOT EXISTS vector`).

### pg_trgm

pg_trgm ships with PostgreSQL in the `postgresql-contrib` package. It is
usually already installed.

```bash
# Ubuntu / Debian — install if not present
sudo apt install postgresql-contrib

# macOS with Homebrew — included by default
```

Like pgvector, the extension is activated by the setup script
(`CREATE EXTENSION IF NOT EXISTS pg_trgm`).

### Create the database

```bash
createdb address_correction
```

## 2. Install Python Dependencies

```bash
cd address
uv sync --all-extras
```

This installs all runtime and dev dependencies into a virtual environment
managed by uv.

## 3. Download Overture Maps Data

Download the Places dataset for the US, Canada, France, and Germany.  This
queries Overture's GeoParquet files on S3 via DuckDB — no API key required.

```bash
uv run python scripts/download_overture.py \
    --output data/overture_places.parquet \
    --countries US CA FR DE
```

The download extracts vendor name, street address, city, postcode, and
country code.  Output is a single Parquet file.

**Expect:** The full download for four countries is several GB and may take
10–30 minutes depending on bandwidth.  For a quick test, use the built-in
sample data (step 4 without `--parquet`).

Options:

```
--output PATH       Output file (default: data/overture_places.parquet)
--countries US FR   ISO codes to include (default: US CA FR DE)
--release TAG       Overture release tag (default: 2024-12-18.0)
```

## 4. Build the Index (Fit Vectorizer + Load PostgreSQL)

This step fits the TF-IDF vectorizer on the reference data, computes
embeddings, creates the pgvector table, and inserts all records.

```bash
# Using downloaded Overture data
uv run python scripts/build_index.py \
    --parquet data/overture_places.parquet \
    --conninfo "dbname=address_correction" \
    --model-dir model

# Or using built-in sample data (28 records, for quick testing)
uv run python scripts/build_index.py \
    --conninfo "dbname=address_correction" \
    --model-dir model
```

The fitted vectorizer (TF-IDF vocabulary + SVD projection matrix) is saved
to `model/`.  This directory is needed at query time.

Options:

```
--parquet PATH      Parquet file from step 3 (omit to use sample data)
--conninfo STR      PostgreSQL connection string
--model-dir PATH    Where to save the fitted vectorizer (default: model/)
--dimensions INT    Embedding dimensions (default: 256)
```

## 5. Run the Demo

Correct 5 OCR-corrupted receipt addresses from real retail locations:

```bash
uv run python scripts/demo_corrections.py \
    --conninfo "dbname=address_correction" \
    --model-dir model
```

Example output:

```
[PASS] US: 8->B, o->0 OCR confusion
  Vendor:    Apple Store
  OCR input: 1B9 The Gr0ve Dr
  Corrected: 189 The Grove Dr
  City:      Los Angeles
  Conf:      0.871
```

## 6. Run Tests

Tests do **not** require PostgreSQL — all database calls are mocked.

```bash
cd address
uv run pytest -v
```

Test suite contents:

| File | What it tests |
|------|---------------|
| `test_vectorizer.py` | TF-IDF + SVD fitting, transform, save/load, similarity properties |
| `test_db.py` | All PostgreSQL functions with mocked connection |
| `test_matcher.py` | Two-stage correction with mocked `db.search_similar` |
| `test_corrections.py` | 5 real retail locations with OCR-corrupted addresses (no DB needed) |

## 7. Use in Your Own Code

```python
from pathlib import Path
from address_correction import AddressMatcher, AddressVectorizer, db

# Load the fitted vectorizer
vectorizer = AddressVectorizer.load(Path("model"))

# Connect to PostgreSQL
conn = db.get_connection("dbname=address_correction")

# Create the matcher
matcher = AddressMatcher(vectorizer, conn)

# Correct an OCR'd address
result = matcher.correct(
    vendor_name="Galeries Lafayette",
    address="40 Bou1evard Haussrnann",
)

if result.matched:
    print(f"Corrected: {result.corrected_address}, {result.corrected_city}")
    print(f"Confidence: {result.confidence:.3f}")
else:
    print("No confident match found")
```

### Tuning Parameters

`AddressMatcher` accepts these keyword arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 20 | Number of vector-search candidates before reranking |
| `confidence_threshold` | 0.45 | Minimum combined score to accept a correction |
| `vendor_weight` | 0.5 | Weight of vendor-name string similarity in final score |
| `address_weight` | 0.3 | Weight of address string similarity in final score |
| `embedding_weight` | 0.2 | Weight of vector cosine similarity in final score |

## Architecture

```
Receipt OCR
    |
    v
(vendor_name, corrupted_address)
    |
    v
+--------------------+
| AddressVectorizer  |  TF-IDF char n-grams (3-5) + SVD
| transform_one()    |  -> dense vector (256-d)
+--------------------+
    |
    v
+--------------------+
| pgvector HNSW      |  Approximate nearest-neighbor search
| db.search_similar  |  -> top-K candidates
+--------------------+
    |
    v
+--------------------+
| Reranker           |  RapidFuzz ratio on vendor name + address
| AddressMatcher     |  -> best match + confidence score
+--------------------+
    |
    v
CorrectionResult
```

## How It Handles OCR Errors

Character n-gram embeddings are inherently robust to OCR corruption because
most n-grams survive even when individual characters are wrong.

Example: `"Apple Store 1B9 The Gr0ve Dr"` vs `"Apple Store 189 The Grove Dr"`

The strings share ~80% of their character 4-grams (`"Appl"`, `"pple"`,
`" Sto"`, `"Stor"`, `"tore"`, `"The "`, ...).  This high overlap produces
a high cosine similarity in embedding space, pulling the correct reference
record to the top of the candidate list.  The reranker then confirms the
match using vendor-name string similarity.
