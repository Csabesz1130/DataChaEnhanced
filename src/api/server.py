from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health as health_router
from src.api.routers import version as version_router

app = FastAPI(title="Signal Analyzer API", version="1.0.0")

# CORS (open by default; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router.router)
app.include_router(version_router.router)

@app.get("/", summary="API root")
def root():
    return {
        "name": "Signal Analyzer API",
        "endpoints": [
            "/health",
            "/version",
        ],
    }