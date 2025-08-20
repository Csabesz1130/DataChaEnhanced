from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover
	np = None  # type: ignore
try:
	import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
	plt = None  # type: ignore

from src.utils.logger import app_logger


@dataclass
class ClusterResult:
	labels: np.ndarray
	n_clusters: int
	silhouette: Optional[float]
	figure_path: Optional[str]


class SelfLearningModule:
	"""
	Nem felügyelt csoportosítás vektortereken, tiszta NumPy megközelítéssel
	(sklearn nélkül), hogy korlátozott környezetben is fusson.
	"""

	def __init__(self, output_dir: str = "/workspace/performance_analysis/performance_reports"):
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

	def _to_ndarray(self, X: List[List[float]]):
		if np is None:
			raise RuntimeError("NumPy szükséges a klaszterezéshez. Kérlek telepítsd a numpy csomagot.")
		return np.asarray(X, dtype=np.float32)

	def _kmeans_numpy(self, X, k: int, iters: int = 100, seed: int = 42):
		if np is None:
			raise RuntimeError("NumPy szükséges a klaszterezéshez. Kérlek telepítsd a numpy csomagot.")
		rng = np.random.default_rng(seed)
		n, d = X.shape
		# k kezdeti centroid véletlen mintákkal
		idx = rng.choice(n, size=min(k, n), replace=False)
		centroids = X[idx].copy()
		labels = np.zeros(n, dtype=np.int32)
		for _ in range(iters):
			# hozzárendelés
			# távolságok: (n,k)
			dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
			new_labels = np.argmin(dists, axis=1)
			if np.array_equal(new_labels, labels):
				break
			labels = new_labels
			# centroid frissítés
			for c in range(centroids.shape[0]):
				members = X[labels == c]
				if len(members) > 0:
					centroids[c] = members.mean(axis=0)
		return labels

	def _silhouette_numpy(self, X, labels) -> Optional[float]:
		if np is None:
			raise RuntimeError("NumPy szükséges a sziluett számításhoz. Kérlek telepítsd a numpy csomagot.")
		# egyszerűsített, O(n^2) sziluett becslés kicsi n esetére
		n = X.shape[0]
		if n < 3:
			return None
		unique = np.unique(labels)
		if unique.shape[0] < 2:
			return None
		# pairwise távolságok
		diffs = X[:, None, :] - X[None, :, :]
		D = np.linalg.norm(diffs, axis=2)
		s_vals = []
		for i in range(n):
			same = labels == labels[i]
			other = labels != labels[i]
			a = np.mean(D[i, same]) if np.sum(same) > 1 else 0.0
			b = np.inf
			for c in unique:
				if c == labels[i]:
					continue
				mask = labels == c
				if np.any(mask):
					b = min(b, float(np.mean(D[i, mask])))
			if not np.isfinite(b):
				continue
			s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
			s_vals.append(s)
		if not s_vals:
			return None
		return float(np.mean(s_vals))

	def _auto_select_k(self, embeddings, max_k: int = 10) -> Tuple[int, Optional[float]]:
		if np is None:
			raise RuntimeError("NumPy szükséges az automatikus klaszterszám kiválasztásához. Kérlek telepítsd a numpy csomagot.")
		if embeddings.shape[0] < 3:
			return (1, None)
		best_k = 2
		best_s: float = -1.0
		for k in range(2, min(max_k, embeddings.shape[0]) + 1):
			labels = self._kmeans_numpy(embeddings, k)
			s = self._silhouette_numpy(embeddings, labels) or -1.0
			if s > best_s:
				best_s = s
				best_k = k
		return (best_k, best_s if best_s >= 0 else None)

	def cluster_embeddings(self, embeddings: List[List[float]], n_clusters: Optional[int] = None, max_k: int = 10, visualize: bool = True) -> ClusterResult:
		if embeddings is None or len(embeddings) == 0:
			raise ValueError("Empty embeddings for clustering")
		X = self._to_ndarray(embeddings)
		if n_clusters is None:
			n_clusters, sil = self._auto_select_k(X, max_k=max_k)
			app_logger.info(f"SelfLearningModule: auto-selected k={n_clusters} (silhouette={sil})")
		else:
			sil = None
		labels = self._kmeans_numpy(X, n_clusters)
		fig_path = None
		if visualize and X.shape[0] >= 2:
			if plt is None:
				app_logger.warning("Matplotlib nem elérhető, vizualizáció kihagyva.")
			else:
				fig_path = self._visualize(X, labels)
		return ClusterResult(labels=labels, n_clusters=n_clusters, silhouette=sil, figure_path=fig_path)

	def _visualize(self, embeddings, labels) -> str:
		if np is None:
			raise RuntimeError("NumPy szükséges a vizualizációhoz. Kérlek telepítsd a numpy csomagot.")
		if plt is None:
			raise RuntimeError("Matplotlib szükséges a vizualizációhoz. Kérlek telepítsd a matplotlib csomagot.")
		# PCA saját implementáció helyett NumPy SVD
		X = embeddings - embeddings.mean(axis=0, keepdims=True)
		U, S, Vt = np.linalg.svd(X, full_matrices=False)
		pts = X @ Vt.T[:, :2]
		import matplotlib.pyplot as plt
		plt.figure(figsize=(6, 5))
		scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", alpha=0.8, s=40)
		plt.title("Document Embeddings - PCA Clusters")
		plt.xlabel("PC1")
		plt.ylabel("PC2")
		plt.grid(True, alpha=0.3)
		handles, _ = scatter.legend_elements()
		plt.legend(handles, [f"Cluster {i}" for i in range(len(handles))], title="Clusters", loc="best")
		out_path = os.path.join(self.output_dir, "clusters.png")
		plt.tight_layout()
		plt.savefig(out_path, dpi=150)
		plt.close()
		app_logger.info(f"SelfLearningModule: cluster visualization saved to {out_path}")
		return out_path
