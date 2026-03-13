"""Tests for the db module with mocked PostgreSQL connection."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from address_correction import db


@pytest.fixture()
def mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


def test_create_tables(mock_conn):
    conn, cursor = mock_conn
    db.create_tables(conn, dimensions=128)

    # Should execute 2x CREATE EXTENSION, CREATE TABLE, and two CREATE INDEX
    assert cursor.execute.call_count == 5
    create_table_sql = cursor.execute.call_args_list[2][0][0]
    assert "vector(128)" in create_table_sql
    conn.commit.assert_called_once()


def test_drop_tables(mock_conn):
    conn, cursor = mock_conn
    db.drop_tables(conn)

    cursor.execute.assert_called_once()
    assert "DROP TABLE" in cursor.execute.call_args[0][0]
    conn.commit.assert_called_once()


def test_insert_references(mock_conn):
    conn, cursor = mock_conn
    records = [
        {"vendor_name": "Test", "address": "123 Main St", "city": "NYC", "postcode": "10001", "country": "US"},
        {"vendor_name": "Test2", "address": "456 Oak Ave", "city": "LA", "postcode": "90001", "country": "US"},
    ]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    db.insert_references(conn, records, embeddings)

    assert cursor.execute.call_count == 2
    conn.commit.assert_called_once()


def test_search_similar(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.return_value = [
        (1, "Apple Store", "189 The Grove Dr", "Los Angeles", "90036", "US", 0.95),
        (2, "Target", "7100 Santa Monica Blvd", "West Hollywood", "90046", "US", 0.72),
    ]

    query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = db.search_similar(conn, query_embedding, top_k=5)

    assert len(results) == 2
    assert results[0]["vendor_name"] == "Apple Store"
    assert results[0]["similarity"] == 0.95
    assert results[1]["vendor_name"] == "Target"

    # Verify the query used the embedding and top_k
    sql = cursor.execute.call_args[0][0]
    assert "<=>" in sql  # cosine distance operator
    params = cursor.execute.call_args[0][1]
    assert params[2] == 5  # top_k


def test_search_similar_empty(mock_conn):
    conn, cursor = mock_conn
    cursor.fetchall.return_value = []

    results = db.search_similar(conn, np.array([0.1, 0.2]), top_k=10)
    assert results == []


@patch("address_correction.db.psycopg")
@patch("address_correction.db.register_vector")
def test_get_connection(mock_register, mock_psycopg):
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    conn = db.get_connection("dbname=test")

    mock_psycopg.connect.assert_called_once_with("dbname=test")
    mock_register.assert_called_once_with(mock_conn)
    assert conn is mock_conn
