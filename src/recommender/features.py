from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class FeatureBundle:
    tfidf: sparse.csr_matrix
    dense: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD


def fit_content_features(
    train_documents: list[str],
    all_documents: list[str],
    max_features: int = 8_000,
    n_components: int = 256,
    seed: int = 42,
) -> FeatureBundle:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train_documents)
    all_tfidf = vectorizer.transform(all_documents).tocsr()
    components = min(n_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=seed)
    svd.fit(train_tfidf)
    dense = normalize(svd.transform(all_tfidf), norm="l2").astype(np.float32)
    return FeatureBundle(tfidf=all_tfidf, dense=dense, vectorizer=vectorizer, svd=svd)
