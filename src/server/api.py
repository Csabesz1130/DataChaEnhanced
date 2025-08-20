from __future__ import annotations

import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.config import load_config
from src.jobs import JobStore, Job, JobStatus
from src.security import detect_true_file_type, is_forbidden_magic
from src.utils.logger import app_logger
from src.processors import ParserNexus
from src.ai.pipeline_runner import run_embedding_and_anomaly


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="Cognitive Document Pipeline API")
    store = JobStore(cfg.jobs_db_path)

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)):
        jid = str(uuid.uuid4())
        # Save raw file
        raw_path = os.path.join(cfg.raw_storage_dir, f"{jid}_{file.filename}")
        content = await file.read()
        with open(raw_path, "wb") as f:
            f.write(content)

        # Security gateway (magic bytes)
        head = content[:16]
        if is_forbidden_magic(head):
            store.upsert(Job(id=jid, filepath=raw_path, status=JobStatus.QUARANTINED_SECURITY_RISK, message="Forbidden file type"))
            return JSONResponse(status_code=202, content={"job_id": jid, "status": JobStatus.QUARANTINED_SECURITY_RISK})

        true_type = detect_true_file_type(head)
        if true_type:
            app_logger.info(f"True file type detected: {true_type}")

        # Simulate background processing inline (fallback to sync processing)
        try:
            store.upsert(Job(id=jid, filepath=raw_path, status=JobStatus.PROCESSING))
            ir = ParserNexus.parse_file(raw_path)
            if not ir.chunks:
                store.upsert(
                    Job(
                        id=jid,
                        filepath=raw_path,
                        status=JobStatus.QUARANTINED_EXTRACTION_FAILED,
                        message="No content extracted",
                    )
                )
            else:
                app_logger.info(f"Ingested {len(ir.chunks)} chunks")
                store.upsert(Job(id=jid, filepath=raw_path, status=JobStatus.DONE, message=f"chunks={len(ir.chunks)}"))
        except Exception as exc:
            store.upsert(Job(id=jid, filepath=raw_path, status=JobStatus.FAILED, message=str(exc)))

        # Optional: lightweight anomaly run on this single file (could be batch in real system)
        try:
            _emb, _lbl = run_embedding_and_anomaly([raw_path])
        except Exception:
            pass

        return JSONResponse(status_code=202, content={"job_id": jid, "status": JobStatus.ACCEPTED})

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        job = store.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return job.__dict__

    return app


