"""Tests for AddressMatcher with mocked db.search_similar."""

from unittest.mock import patch

import numpy as np
import pytest

from address_correction.matcher import AddressMatcher, CorrectionResult
from address_correction.vectorizer import AddressVectorizer


@pytest.fixture()
def matcher(fitted_vectorizer):
    # conn is unused because we mock db.search_similar
    return AddressMatcher(fitted_vectorizer, conn=None)


def _make_candidate(vendor, address, city, postcode, country, similarity):
    return {
        "id": 1,
        "vendor_name": vendor,
        "address": address,
        "city": city,
        "postcode": postcode,
        "country": country,
        "similarity": similarity,
    }


@patch("address_correction.matcher.db.search_similar")
def test_correct_finds_match(mock_search, matcher):
    mock_search.return_value = [
        _make_candidate("Apple Store", "189 The Grove Dr", "Los Angeles", "90036", "US", 0.92),
        _make_candidate("Target", "7100 Santa Monica Blvd", "West Hollywood", "90046", "US", 0.60),
    ]

    result = matcher.correct("Apple Store", "1B9 The Gr0ve Dr")

    assert result.matched is True
    assert result.corrected_address == "189 The Grove Dr"
    assert result.corrected_city == "Los Angeles"
    assert result.confidence > 0.45


@patch("address_correction.matcher.db.search_similar")
def test_correct_no_candidates(mock_search, matcher):
    mock_search.return_value = []

    result = matcher.correct("Unknown Vendor", "999 Nowhere Rd")

    assert result.matched is False
    assert result.corrected_address is None
    assert result.confidence == 0.0


@patch("address_correction.matcher.db.search_similar")
def test_correct_low_confidence(mock_search, matcher):
    mock_search.return_value = [
        _make_candidate("Completely Different", "999 Other Rd", "Nowhere", "00000", "US", 0.10),
    ]

    result = matcher.correct("Apple Store", "189 The Grove Dr")

    assert result.matched is False


@patch("address_correction.matcher.db.search_similar")
def test_vendor_name_drives_ranking(mock_search, matcher):
    """When two candidates have similar addresses, the matching vendor should win."""
    mock_search.return_value = [
        _make_candidate("Wrong Store", "189 The Grove Dr", "Los Angeles", "90036", "US", 0.90),
        _make_candidate("Apple Store", "189 The Grove Dr", "Los Angeles", "90036", "US", 0.85),
    ]

    result = matcher.correct("Apple Store", "189 The Grove Dr")

    assert result.matched is True
    assert result.corrected_address == "189 The Grove Dr"


@patch("address_correction.matcher.db.search_similar")
def test_result_dataclass_fields(mock_search, matcher):
    mock_search.return_value = [
        _make_candidate("KaDeWe", "Tauentzienstrasse 21-24", "Berlin", "10789", "DE", 0.88),
    ]

    result = matcher.correct("KaDeWe", "Tauentzienstra8e 2l-24")

    assert isinstance(result, CorrectionResult)
    assert result.original_vendor == "KaDeWe"
    assert result.original_address == "Tauentzienstra8e 2l-24"
    assert result.corrected_country == "DE"
