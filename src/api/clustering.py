from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def compute_embedding(
    features: pd.DataFrame,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    X = features.fillna(0.0).to_numpy(dtype=float)
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(X)
    # default PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


def cluster_points(
    embedding: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    labels: np.ndarray
    metrics: Dict[str, float] = {}

    if algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embedding)
    else:
        model = KMeans(n_clusters=max(2, int(n_clusters)), n_init=10)
        labels = model.fit_predict(embedding)

    # silhouette only if >1 cluster and not all noise (-1)
    unique = np.unique(labels)
    if len(unique) > 1 and not (len(unique) == 1 and unique[0] == -1):
        try:
            metrics["silhouette"] = float(silhouette_score(embedding, labels))
        except Exception:
            pass

    return labels, metrics