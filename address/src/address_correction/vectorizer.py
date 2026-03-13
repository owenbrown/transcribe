from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class AddressVectorizer:
    """Converts (vendor_name, address) pairs into dense vectors using
    character n-gram TF-IDF with SVD dimensionality reduction.

    The combined vendor+address string is decomposed into overlapping character
    n-grams (3 to 5 characters by default), weighted by TF-IDF, then projected
    to a dense vector via truncated SVD. The resulting vectors are stored in
    pgvector for approximate nearest-neighbor search.
    """

    def __init__(self, n_components: int = 256, ngram_range: tuple[int, int] = (3, 5)):
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=50_000,
            lowercase=True,
            strip_accents="unicode",
        )
        self.svd: TruncatedSVD | None = None
        self._is_fitted = False

    @staticmethod
    def _prepare_text(vendor_name: str, address: str) -> str:
        return f"{vendor_name} {address}".lower()

    @property
    def dimensions(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before accessing dimensions")
        assert self.svd is not None
        return self.svd.n_components

    def fit(self, vendor_names: list[str], addresses: list[str]) -> "AddressVectorizer":
        texts = [self._prepare_text(v, a) for v, a in zip(vendor_names, addresses)]
        tfidf_matrix = self.tfidf.fit_transform(texts)

        # SVD components cannot exceed the matrix rank
        max_components = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1]) - 1
        actual_components = min(self.n_components, max(1, max_components))

        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._is_fitted = True
        return self

    def transform(self, vendor_names: list[str], addresses: list[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")
        texts = [self._prepare_text(v, a) for v, a in zip(vendor_names, addresses)]
        tfidf_matrix = self.tfidf.transform(texts)
        return self.svd.transform(tfidf_matrix).astype(np.float32)

    def transform_one(self, vendor_name: str, address: str) -> np.ndarray:
        return self.transform([vendor_name], [address])[0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf, path / "tfidf.joblib")
        joblib.dump(self.svd, path / "svd.joblib")

    @classmethod
    def load(cls, path: Path) -> "AddressVectorizer":
        vec = cls.__new__(cls)
        vec.tfidf = joblib.load(path / "tfidf.joblib")
        vec.svd = joblib.load(path / "svd.joblib")
        vec.ngram_range = vec.tfidf.ngram_range
        vec.n_components = vec.svd.n_components
        vec._is_fitted = True
        return vec
