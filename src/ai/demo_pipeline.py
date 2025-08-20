from __future__ import annotations

import os
import sys
import glob
from typing import List

from src.utils.logger import app_logger
from src.processors import ParserNexus
from src.ai import CognitiveCore, SelfLearningModule


def run_demo(filepaths: List[str]) -> None:
    app_logger.info("Demo: Parsing files via ParserNexus")
    irs = []
    for fp in filepaths:
        if not os.path.exists(fp):
            app_logger.warning(f"File not found: {fp}")
            continue
        ir = ParserNexus.parse_file(fp)
        app_logger.info(f"Parsed {fp} into {len(ir.chunks)} chunks")
        irs.append(ir)
    texts = ParserNexus.flatten_texts(irs)
    if not texts:
        app_logger.warning("No texts extracted; aborting demo.")
        return

    app_logger.info("Demo: Building embeddings")
    core = CognitiveCore()
    core.fit_corpus(texts)
    core.build_index()

    app_logger.info("Demo: Running example search")
    for q in [
        "action potential current trace",
        "invoice total and due date",
        "experimental protocol",
    ]:
        results = core.search(q, top_k=3)
        app_logger.info(f"Query: {q}")
        for r in results:
            app_logger.info(f"  -> idx={r.index}, score={r.score:.3f}, text='{texts[r.index][:120]}'")

    try:
        app_logger.info("Demo: Unsupervised clustering")
        sl = SelfLearningModule()
        result = sl.cluster_embeddings(core.get_embeddings(), visualize=True)
        app_logger.info(
            f"Clusters: k={result.n_clusters}, silhouette={result.silhouette}, figure={result.figure_path}"
        )
    except Exception as exc:
        app_logger.warning(f"Klaszterezés kihagyva: {exc}")


if __name__ == "__main__":
    # CLI használat: python demo_pipeline.py <file|dir|glob> [...]
    args = sys.argv[1:]
    candidates: List[str] = []
    if not args:
        candidates = [
            "/workspace/data/202304_0521.atf",
            "/workspace/data/202304_0523.atf",
        ]
    else:
        exts = ["*.xlsx", "*.xls", "*.csv", "*.pdf", "*.docx", "*.atf", "*.txt"]
        for a in args:
            p = a.strip().strip('"')
            if os.path.isdir(p):
                for pat in exts:
                    candidates.extend(glob.glob(os.path.join(p, pat)))
            elif any(ch in p for ch in ["*", "?", "["]):
                candidates.extend(glob.glob(p))
            else:
                candidates.append(p)
    run_demo(candidates)


