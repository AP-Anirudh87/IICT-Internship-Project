"""
========================================================
  Fake News Detection – Feature Extractor
  IICT Summer Internship Project 1
========================================================
Implements two feature representations from scratch
AND wraps sklearn's versions for comparison:

  1. ManualBoW   – Bag-of-Words (count vectors)
  2. ManualTFIDF – TF-IDF vectors (manual math)
  3. SklearnTFIDF – sklearn TfidfVectorizer wrapper

Both manual implementations are vectorized with numpy
for efficiency but avoid sklearn's sparse machinery so
the math is fully transparent.
"""

import math
import numpy as np
from collections import Counter
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  1. Manual Bag-of-Words
# ══════════════════════════════════════════════════════════════════════════════
class ManualBoW:
    """
    Manual Bag-of-Words vectorizer.

    After calling fit(), vocabulary_ maps each token → column index.
    transform() returns a dense numpy array of shape (n_docs, vocab_size).

    Parameters
    ----------
    max_features : int or None
        Keep only the top-k most frequent tokens.
    """

    def __init__(self, max_features: Optional[int] = 5000) -> None:
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, documents: List[str]) -> "ManualBoW":
        """Build vocabulary from a list of pre-tokenized strings."""
        counter: Counter = Counter()
        for doc in documents:
            counter.update(doc.split())

        if self.max_features:
            most_common = counter.most_common(self.max_features)
        else:
            most_common = counter.most_common()

        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Return count matrix, shape (n_docs, vocab_size)."""
        if not self.vocabulary_:
            raise RuntimeError("Call fit() before transform().")
        n     = len(documents)
        vocab = len(self.vocabulary_)
        matrix = np.zeros((n, vocab), dtype=np.float32)
        for i, doc in enumerate(documents):
            for token in doc.split():
                col = self.vocabulary_.get(token)
                if col is not None:
                    matrix[i, col] += 1
        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Manual TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
class ManualTFIDF:
    """
    Manual TF-IDF vectorizer (smooth IDF, L2 normalisation).

    TF  = count(t, d) / len(d)
    IDF = log((1 + N) / (1 + df(t))) + 1      [sklearn-compatible smooth IDF]

    Parameters
    ----------
    max_features : int or None
    """

    def __init__(self, max_features: Optional[int] = 10000) -> None:
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._n_docs: int = 0

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, documents: List[str]) -> "ManualTFIDF":
        N = len(documents)
        self._n_docs = N

        # Document frequency
        df: Counter = Counter()
        for doc in documents:
            unique_tokens = set(doc.split())
            df.update(unique_tokens)

        # Select vocabulary by document frequency
        if self.max_features:
            selected = df.most_common(self.max_features)
        else:
            selected = list(df.items())

        self.vocabulary_ = {w: i for i, (w, _) in enumerate(selected)}

        # Compute IDF for each vocabulary word
        idf_vals = []
        for word, _ in selected:
            df_t = df[word]
            idf  = math.log((1 + N) / (1 + df_t)) + 1
            idf_vals.append(idf)
        self._idf = np.array(idf_vals, dtype=np.float64)
        return self

    # ── Transform ────────────────────────────────────────────────────────────
    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise RuntimeError("Call fit() before transform().")
        n      = len(documents)
        vocab  = len(self.vocabulary_)
        matrix = np.zeros((n, vocab), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = doc.split()
            if not tokens:
                continue
            tf_counts: Counter = Counter(tokens)
            for token, count in tf_counts.items():
                col = self.vocabulary_.get(token)
                if col is not None:
                    tf = count / len(tokens)
                    matrix[i, col] = tf * self._idf[col]

            # L2 normalise each row
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Sklearn TF-IDF wrapper (for comparison)
# ══════════════════════════════════════════════════════════════════════════════
class SklearnTFIDF:
    """
    Thin wrapper around sklearn TfidfVectorizer that returns a dense matrix
    with the same interface as ManualTFIDF.
    """

    def __init__(self, max_features: int = 10000, **kwargs) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=False,     # keep consistent with manual version
            **kwargs
        )

    def fit(self, documents: List[str]) -> "SklearnTFIDF":
        self._vectorizer.fit(documents)
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        return self._vectorizer.transform(documents).toarray()

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self._vectorizer.fit_transform(documents).toarray()

    @property
    def vocabulary_(self) -> Dict[str, int]:
        return self._vectorizer.vocabulary_


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        "election fraud scandal politician votes",
        "election results president win polls",
        "cat dog pet animal food water",
        "breaking news fake story media propaganda",
    ]

    print("=" * 60)
    print("  Manual Bag-of-Words")
    print("=" * 60)
    bow = ManualBoW(max_features=10)
    X_bow = bow.fit_transform(docs)
    print(f"Shape: {X_bow.shape}")
    print(f"Vocab: {list(bow.vocabulary_.keys())[:8]} …")

    print("\n" + "=" * 60)
    print("  Manual TF-IDF")
    print("=" * 60)
    tfidf_manual = ManualTFIDF(max_features=10)
    X_manual = tfidf_manual.fit_transform(docs)
    print(f"Shape: {X_manual.shape}")
    print(f"Doc-0 top scores: {sorted(X_manual[0], reverse=True)[:5]}")

    print("\n" + "=" * 60)
    print("  Sklearn TF-IDF")
    print("=" * 60)
    tfidf_sk = SklearnTFIDF(max_features=10)
    X_sk = tfidf_sk.fit_transform(docs)
    print(f"Shape: {X_sk.shape}")
    print(f"Doc-0 top scores: {sorted(X_sk[0], reverse=True)[:5]}")
