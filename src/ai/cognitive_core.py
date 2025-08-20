from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from src.utils.logger import app_logger


@dataclass
class RetrievalResult:
    index: int
    score: float


class _BasicVectorizer:
    """
    Egyszerű TF-IDF vektorizáló tiszta NumPy/Python megvalósítással
    (sklearn nélküli környezethez).
    """

    def __init__(self):
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        # nagyon egyszerű tokenizálás: kisbetű, nem alfanumerikus szűrés
        buf = []
        word = []
        for ch in text.lower():
            if ch.isalnum() or ch in ["_", "-"]:
                word.append(ch)
            else:
                if word:
                    buf.append("".join(word))
                    word = []
        if word:
            buf.append("".join(word))
        return buf

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # építsük fel a vocab-ot és a dokumentum gyakoriságokat
        doc_freq: Dict[str, int] = {}
        tokenized_docs: List[List[str]] = []
        for t in texts:
            tokens = self._tokenize(t)
            tokenized_docs.append(tokens)
            seen = set()
            for tok in tokens:
                if tok not in self.vocabulary_:
                    self.vocabulary_[tok] = len(self.vocabulary_)
                if tok not in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
                    seen.add(tok)
        n_docs = len(texts)
        vocab_size = len(self.vocabulary_)
        idf: List[float] = [0.0] * vocab_size
        for tok, idx in self.vocabulary_.items():
            df = doc_freq.get(tok, 1)
            idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        # store as list, but expose numpy-like API via our functions
        import array as _array  # local import to avoid global deps
        self.idf_ = _array.array('f', idf)
        # számoljuk ki a TF-IDF mátrixot
        mat: List[List[float]] = [[0.0] * vocab_size for _ in range(n_docs)]
        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf: Dict[int, int] = {}
            for tok in tokens:
                idx = self.vocabulary_[tok]
                tf[idx] = tf.get(idx, 0) + 1
            max_tf = max(tf.values()) if tf else 1
            for idx, count in tf.items():
                tf_weight = 0.5 + 0.5 * (count / max_tf)  # augmented TF
                mat[i][idx] = tf_weight * float(self.idf_[idx])
        # normalizáljuk L2-re
        for i in range(n_docs):
            s = sum(v * v for v in mat[i])
            norm = math.sqrt(s) + 1e-12
            mat[i] = [v / norm for v in mat[i]]
        return mat

    def transform(self, texts: List[str]) -> List[List[float]]:
        if self.idf_ is None or not self.vocabulary_:
            raise ValueError("Vectorizer not fitted")
        vocab_size = len(self.vocabulary_)
        mat: List[List[float]] = [[0.0] * vocab_size for _ in texts]
        for i, t in enumerate(texts):
            tokens = self._tokenize(t)
            if not tokens:
                continue
            tf: Dict[int, int] = {}
            for tok in tokens:
                idx = self.vocabulary_.get(tok)
                if idx is None:
                    continue
                tf[idx] = tf.get(idx, 0) + 1
            max_tf = max(tf.values()) if tf else 1
            for idx, count in tf.items():
                tf_weight = 0.5 + 0.5 * (count / max_tf)
                mat[i][idx] = tf_weight * float(self.idf_[idx])
        for i in range(len(texts)):
            s = sum(v * v for v in mat[i])
            norm = math.sqrt(s) + 1e-12
            mat[i] = [v / norm for v in mat[i]]
        return mat


class CognitiveCore:
    """
    Szemantikai vektorizációs mag, több szintű visszaeséssel:
    1) sentence-transformers, ha elérhető
    2) saját TF-IDF (sklearn nélkül)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._basic_vec: Optional[_BasicVectorizer] = None
        self._embeddings: Optional[List[List[float]]] = None
        self._use_sentence_transformers = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(self.model_name)
            self._use_sentence_transformers = True
            app_logger.info(
                f"CognitiveCore: sentence-transformers modell használata: '{self.model_name}'"
            )
        except Exception as exc:
            app_logger.warning(
                f"CognitiveCore: sentence-transformers nem elérhető ({exc}). Saját TF-IDF lesz használva."
            )
            self._basic_vec = _BasicVectorizer()

    def _embed_with_st(self, texts: List[str]) -> List[List[float]]:
        assert self._st_model is not None
        vectors = self._st_model.encode(texts, normalize_embeddings=True)
        try:
            return vectors.tolist()
        except Exception:
            # already a list
            return vectors

    def _embed_with_basic(self, texts: List[str], fit: bool) -> List[List[float]]:
        assert self._basic_vec is not None
        if fit:
            return self._basic_vec.fit_transform(texts)
        return self._basic_vec.transform(texts)

    def fit_corpus(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("No texts provided to fit_corpus")
        if self._use_sentence_transformers:
            self._embeddings = self._embed_with_st(texts)
        else:
            self._embeddings = self._embed_with_basic(texts, fit=True)
        dim = len(self._embeddings[0]) if self._embeddings else 0
        app_logger.info(f"CognitiveCore: {len(texts)} darab chunk beágyazva; dim={dim}")
        return self._embeddings

    def build_index(self) -> None:
        if self._embeddings is None:
            raise ValueError("Call fit_corpus first")
        # nincs külön index; a keresés brute-force koszinusz hasonlósággal
        app_logger.info("CognitiveCore: bruteforce koszinusz kereső használva")

    def _embed_query(self, query: str) -> List[List[float]]:
        if self._use_sentence_transformers:
            vec = self._embed_with_st([query])
        else:
            vec = self._embed_with_basic([query], fit=False)
        return vec

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self._embeddings is None:
            raise ValueError("No embeddings. Call fit_corpus first")
        qv = self._embed_query(query)[0]
        # mivel L2-normalizált, a dot = koszinusz hasonlóság
        sims: List[float] = []
        for emb in self._embeddings:
            sims.append(sum(e * q for e, q in zip(emb, qv)))
        n = len(sims)
        top_k = min(top_k, n)
        # top-k indexek kiválasztása
        inds = sorted(range(n), key=lambda i: sims[i], reverse=True)[:top_k]
        return [RetrievalResult(index=i, score=float(sims[i])) for i in inds]

    def get_embeddings(self) -> List[List[float]]:
        if self._embeddings is None:
            raise ValueError("No embeddings. Call fit_corpus first")
        return self._embeddings


