import numpy as np
import pytest

from address_correction.sample_data import REFERENCE_DATA
from address_correction.vectorizer import AddressVectorizer


@pytest.fixture()
def sample_records() -> list[dict]:
    return list(REFERENCE_DATA)


@pytest.fixture()
def fitted_vectorizer(sample_records: list[dict]) -> AddressVectorizer:
    vec = AddressVectorizer(n_components=256)
    vendors = [r["vendor_name"] for r in sample_records]
    addresses = [r["address"] for r in sample_records]
    vec.fit(vendors, addresses)
    return vec


@pytest.fixture()
def reference_embeddings(
    fitted_vectorizer: AddressVectorizer, sample_records: list[dict]
) -> np.ndarray:
    vendors = [r["vendor_name"] for r in sample_records]
    addresses = [r["address"] for r in sample_records]
    return fitted_vectorizer.transform(vendors, addresses)
