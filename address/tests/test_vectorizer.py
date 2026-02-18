import numpy as np
import pytest

from address_correction.vectorizer import AddressVectorizer


def test_fit_and_transform(fitted_vectorizer, sample_records):
    vendors = [r["vendor_name"] for r in sample_records]
    addresses = [r["address"] for r in sample_records]
    embeddings = fitted_vectorizer.transform(vendors, addresses)
    assert embeddings.shape[0] == len(sample_records)
    assert embeddings.shape[1] == fitted_vectorizer.dimensions
    assert embeddings.dtype == np.float32


def test_transform_one(fitted_vectorizer):
    vec = fitted_vectorizer.transform_one("Apple Store", "189 The Grove Dr")
    assert vec.shape == (fitted_vectorizer.dimensions,)
    assert vec.dtype == np.float32


def test_transform_before_fit_raises():
    vec = AddressVectorizer()
    with pytest.raises(RuntimeError, match="fitted"):
        vec.transform(["x"], ["y"])


def test_dimensions_before_fit_raises():
    vec = AddressVectorizer()
    with pytest.raises(RuntimeError, match="fitted"):
        _ = vec.dimensions


def test_similar_inputs_produce_close_vectors(fitted_vectorizer):
    """OCR-corrupted text should have high cosine similarity to the original."""
    original = fitted_vectorizer.transform_one("Apple Store", "189 The Grove Dr")
    corrupted = fitted_vectorizer.transform_one("Apple Store", "1B9 The Gr0ve Dr")

    cosine_sim = float(
        np.dot(original, corrupted)
        / (np.linalg.norm(original) * np.linalg.norm(corrupted))
    )
    assert cosine_sim > 0.7, f"Expected similarity > 0.7, got {cosine_sim:.3f}"


def test_different_vendors_produce_distant_vectors(fitted_vectorizer):
    """Completely different vendors should have lower similarity."""
    apple = fitted_vectorizer.transform_one("Apple Store", "189 The Grove Dr")
    kadewe = fitted_vectorizer.transform_one("KaDeWe", "Tauentzienstrasse 21-24")

    cosine_sim = float(
        np.dot(apple, kadewe)
        / (np.linalg.norm(apple) * np.linalg.norm(kadewe))
    )
    assert cosine_sim < 0.7, f"Expected similarity < 0.7, got {cosine_sim:.3f}"


def test_save_and_load(fitted_vectorizer, tmp_path):
    original_vec = fitted_vectorizer.transform_one("Test", "123 Main St")

    fitted_vectorizer.save(tmp_path / "model")
    loaded = AddressVectorizer.load(tmp_path / "model")

    loaded_vec = loaded.transform_one("Test", "123 Main St")
    np.testing.assert_allclose(original_vec, loaded_vec, atol=1e-6)
    assert loaded.dimensions == fitted_vectorizer.dimensions


def test_dimensions_capped_by_data_size():
    """When n_components exceeds the data, SVD should reduce automatically."""
    vec = AddressVectorizer(n_components=999)
    vec.fit(["A", "B", "C"], ["x", "y", "z"])
    assert vec.dimensions < 999
    assert vec.dimensions >= 1
