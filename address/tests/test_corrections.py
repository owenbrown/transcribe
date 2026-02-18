"""End-to-end correction tests for 5 real retail locations.

These tests verify the core algorithm (TF-IDF char n-gram embedding +
cosine similarity + reranking) without requiring PostgreSQL.  The vector
search is performed directly with numpy instead of pgvector.
"""

import numpy as np
import pytest
from rapidfuzz import fuzz

from address_correction.sample_data import REFERENCE_DATA, TEST_CASES
from address_correction.vectorizer import AddressVectorizer


# ── Helpers ──────────────────────────────────────────────────────────


def _cosine_similarity_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between one query vector and a matrix."""
    query_norm = query / np.linalg.norm(query)
    ref_norms = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    return ref_norms @ query_norm


def _rerank(
    vendor_name: str,
    address: str,
    candidate_indices: np.ndarray,
    similarities: np.ndarray,
    records: list[dict],
) -> tuple[int, float]:
    """Rerank candidates using string similarity. Returns (best_index, score)."""
    best_idx, best_score = -1, -1.0
    for idx in candidate_indices:
        cand = records[idx]
        vendor_sim = fuzz.ratio(vendor_name.lower(), cand["vendor_name"].lower()) / 100.0
        addr_sim = fuzz.ratio(address.lower(), cand["address"].lower()) / 100.0
        emb_sim = float(similarities[idx])
        combined = 0.5 * vendor_sim + 0.3 * addr_sim + 0.2 * emb_sim
        if combined > best_score:
            best_score = combined
            best_idx = int(idx)
    return best_idx, best_score


def _correct(
    vectorizer: AddressVectorizer,
    reference_embeddings: np.ndarray,
    records: list[dict],
    vendor_name: str,
    ocr_address: str,
    top_k: int = 10,
) -> tuple[dict, float]:
    """Run the full correction pipeline (vector retrieval + rerank)."""
    query_vec = vectorizer.transform_one(vendor_name, ocr_address)
    similarities = _cosine_similarity_matrix(query_vec, reference_embeddings)
    top_indices = np.argsort(-similarities)[:top_k]
    best_idx, score = _rerank(vendor_name, ocr_address, top_indices, similarities, records)
    return records[best_idx], score


# ── Test cases ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[c["vendor_name"] for c in TEST_CASES],
)
def test_ocr_correction(
    fitted_vectorizer: AddressVectorizer,
    reference_embeddings: np.ndarray,
    sample_records: list[dict],
    case: dict,
):
    match, score = _correct(
        fitted_vectorizer,
        reference_embeddings,
        sample_records,
        case["vendor_name"],
        case["ocr_address"],
    )
    assert match["vendor_name"] == case["vendor_name"], (
        f"[{case['description']}] Expected vendor '{case['vendor_name']}', "
        f"got '{match['vendor_name']}'"
    )
    assert match["address"] == case["expected_address"], (
        f"[{case['description']}] Expected address '{case['expected_address']}', "
        f"got '{match['address']}'"
    )
    assert match["city"] == case["expected_city"], (
        f"[{case['description']}] Expected city '{case['expected_city']}', "
        f"got '{match['city']}'"
    )
    assert score > 0.45, (
        f"[{case['description']}] Confidence {score:.3f} below threshold"
    )


def test_embedding_similarity_higher_for_correct_match(
    fitted_vectorizer: AddressVectorizer,
    reference_embeddings: np.ndarray,
    sample_records: list[dict],
):
    """The correct vendor+address should always rank in the top 3 by embedding alone."""
    for case in TEST_CASES:
        query_vec = fitted_vectorizer.transform_one(case["vendor_name"], case["ocr_address"])
        similarities = _cosine_similarity_matrix(query_vec, reference_embeddings)
        top3_indices = np.argsort(-similarities)[:3]
        top3_vendors = [sample_records[i]["vendor_name"] for i in top3_indices]
        assert case["vendor_name"] in top3_vendors, (
            f"[{case['description']}] '{case['vendor_name']}' not in top-3 "
            f"by embedding similarity. Got: {top3_vendors}"
        )
