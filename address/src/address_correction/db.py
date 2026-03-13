"""PostgreSQL + pgvector database functions.

Every function that touches the database lives here so that tests can mock
this module cleanly.
"""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def get_connection(conninfo: str) -> psycopg.Connection:
    conn = psycopg.connect(conninfo)
    register_vector(conn)
    return conn


def create_tables(conn: psycopg.Connection, dimensions: int = 256) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS address_references (
                id SERIAL PRIMARY KEY,
                vendor_name TEXT NOT NULL,
                address TEXT NOT NULL,
                city TEXT,
                postcode TEXT,
                country TEXT,
                embedding vector({dimensions})
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_address_embedding
            ON address_references
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vendor_name
            ON address_references USING gin (vendor_name gin_trgm_ops)
            """
        )
    conn.commit()


def drop_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS address_references")
    conn.commit()


def insert_references(
    conn: psycopg.Connection,
    records: list[dict],
    embeddings: np.ndarray,
) -> None:
    with conn.cursor() as cur:
        for rec, emb in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO address_references
                    (vendor_name, address, city, postcode, country, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    rec["vendor_name"],
                    rec["address"],
                    rec.get("city"),
                    rec.get("postcode"),
                    rec.get("country"),
                    emb,
                ),
            )
    conn.commit()


def search_similar(
    conn: psycopg.Connection,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, vendor_name, address, city, postcode, country,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM address_references
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding.tolist(), query_embedding.tolist(), top_k),
        )
        columns = [
            "id", "vendor_name", "address", "city", "postcode", "country", "similarity",
        ]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
