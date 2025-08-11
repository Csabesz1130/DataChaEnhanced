from typing import Dict, List, Tuple
import numpy as np


def select_informative_samples(
    embedding: np.ndarray,
    labels: np.ndarray,
    k: int = 20,
) -> List[int]:
    """
    Válasszunk kérdéses pontokat: közel legyen két különböző klaszter középvonalához.
    Egyszerű heurisztika: számoljuk a két legközelebbi szomszéd távolságkülönbségét.
    A legkisebb margin értékű pontok a leginkább bizonytalanok.
    """
    if embedding is None or len(embedding) == 0:
        return []

    # kNN alapú margin
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(10, len(embedding)))
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    # margin: d2 - d1 (minél kisebb annál bizonytalanabb)
    first = distances[:, 1]
    second = distances[:, 2] if distances.shape[1] > 2 else distances[:, 1]
    margin = second - first

    # Válasszuk a legkisebb margin értékű indexeket
    order = np.argsort(margin)
    k = int(min(k, len(order)))
    return [int(i) for i in order[:k]]