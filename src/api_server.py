from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.api.feature_extraction import (
    batch_extract_features,
    list_atf_files,
    load_signal,
)
from src.api.clustering import compute_embedding, cluster_points
from src.api.active_learning import select_informative_samples
from src.api.state_store import LabelStore
from src.api.utils import list_data_files, resolve_path


app = Flask(__name__)
CORS(app)

label_store = LabelStore(path=str(Path("analysis_history.json").resolve()))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/files")
def files_list():
    root = request.args.get("root")
    if root:
        root = resolve_path(root)
        files = list_atf_files(root)
    else:
        files = list_data_files()
    return jsonify({"files": files})


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".atf"):
        return jsonify({"error": "Only .atf is allowed"}), 400

    data_dir = Path("data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / Path(f.filename).name
    f.save(str(dest))
    return jsonify({"saved": str(dest)})


@app.get("/api/signal")
def get_signal():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Missing path"}), 400
    p = resolve_path(path)
    t, y = load_signal(p)
    return jsonify({
        "path": p,
        "time_s": t.tolist(),
        "current_pA": y.tolist(),
    })


@app.post("/api/features")
def features():
    body = request.get_json(silent=True) or {}
    paths: List[str] = body.get("paths", [])
    if not paths:
        return jsonify({"error": "paths required"}), 400
    abs_paths = [resolve_path(p) for p in paths]
    df = batch_extract_features(abs_paths)
    return jsonify({"features": df.fillna(0.0).to_dict(orient="index")})


@app.post("/api/cluster")
def cluster():
    body = request.get_json(silent=True) or {}
    # either provide features object or paths
    features_obj: Dict[str, Dict[str, float]] = body.get("features") or {}
    paths: List[str] = body.get("paths") or list(features_obj.keys())

    if features_obj:
        import pandas as pd
        df = pd.DataFrame.from_dict(features_obj, orient="index")
        df.index.name = "path"
    else:
        if not paths:
            return jsonify({"error": "paths or features required"}), 400
        import pandas as pd
        abs_paths = [resolve_path(p) for p in paths]
        df = batch_extract_features(abs_paths)

    method = body.get("embed_method", "pca")
    algorithm = body.get("algorithm", "kmeans")
    n_clusters = int(body.get("n_clusters", 3))
    eps = float(body.get("eps", 0.5))
    min_samples = int(body.get("min_samples", 5))

    embed = compute_embedding(df, method=method, n_components=2)
    labels, metrics = cluster_points(embed, algorithm=algorithm, n_clusters=n_clusters, eps=eps, min_samples=min_samples)

    return jsonify({
        "paths": df.index.tolist(),
        "embedding": embed.tolist(),
        "labels": [int(x) for x in labels.tolist()],
        "metrics": metrics,
    })


@app.post("/api/active-learning/query")
def al_query():
    body = request.get_json(silent=True) or {}
    embedding = body.get("embedding")
    labels = body.get("labels")
    k = int(body.get("k", 20))
    if embedding is None or labels is None:
        return jsonify({"error": "embedding and labels required"}), 400
    import numpy as np
    idxs = select_informative_samples(np.asarray(embedding), np.asarray(labels), k=k)
    return jsonify({"indices": idxs})


@app.post("/api/labels")
def labels():
    body = request.get_json(silent=True) or {}
    labels_map: Dict[str, str] = body.get("labels", {})
    if not isinstance(labels_map, dict) or not labels_map:
        return jsonify({"error": "labels dict required"}), 400
    labels_abs = {resolve_path(k): str(v) for k, v in labels_map.items()}
    label_store.set_many(labels_abs)
    return jsonify({"saved": len(labels_abs)})


@app.get("/api/labels")
def get_labels():
    return jsonify({"labels": label_store.all()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API server for interactive clustering")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)