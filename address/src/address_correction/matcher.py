from dataclasses import dataclass

from rapidfuzz import fuzz

from . import db
from .vectorizer import AddressVectorizer


@dataclass
class CorrectionResult:
    original_vendor: str
    original_address: str
    corrected_address: str | None
    corrected_city: str | None
    corrected_postcode: str | None
    corrected_country: str | None
    confidence: float
    matched: bool


class AddressMatcher:
    """Two-stage address correction: vector retrieval then string-similarity reranking."""

    def __init__(
        self,
        vectorizer: AddressVectorizer,
        conn,
        *,
        top_k: int = 20,
        confidence_threshold: float = 0.45,
        vendor_weight: float = 0.5,
        address_weight: float = 0.3,
        embedding_weight: float = 0.2,
    ):
        self.vectorizer = vectorizer
        self.conn = conn
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.vendor_weight = vendor_weight
        self.address_weight = address_weight
        self.embedding_weight = embedding_weight

    def correct(self, vendor_name: str, address: str) -> CorrectionResult:
        # Stage 1: vector retrieval
        query_vec = self.vectorizer.transform_one(vendor_name, address)
        candidates = db.search_similar(self.conn, query_vec, top_k=self.top_k)

        if not candidates:
            return self._no_match(vendor_name, address)

        # Stage 2: rerank by string similarity
        best, best_score = None, -1.0
        for cand in candidates:
            score = self._score(vendor_name, address, cand)
            if score > best_score:
                best_score = score
                best = cand

        if best is None or best_score < self.confidence_threshold:
            return self._no_match(vendor_name, address)

        return CorrectionResult(
            original_vendor=vendor_name,
            original_address=address,
            corrected_address=best["address"],
            corrected_city=best.get("city"),
            corrected_postcode=best.get("postcode"),
            corrected_country=best.get("country"),
            confidence=best_score,
            matched=True,
        )

    def _score(self, vendor_name: str, address: str, candidate: dict) -> float:
        vendor_sim = fuzz.ratio(vendor_name.lower(), candidate["vendor_name"].lower()) / 100.0
        address_sim = fuzz.ratio(address.lower(), candidate["address"].lower()) / 100.0
        embedding_sim = max(0.0, candidate.get("similarity", 0.0))
        return (
            self.vendor_weight * vendor_sim
            + self.address_weight * address_sim
            + self.embedding_weight * embedding_sim
        )

    @staticmethod
    def _no_match(vendor_name: str, address: str) -> CorrectionResult:
        return CorrectionResult(
            original_vendor=vendor_name,
            original_address=address,
            corrected_address=None,
            corrected_city=None,
            corrected_postcode=None,
            corrected_country=None,
            confidence=0.0,
            matched=False,
        )
