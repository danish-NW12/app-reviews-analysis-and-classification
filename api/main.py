"""
FastAPI application for app reviews analysis dashboard API.

Run from project root:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure project root is on path for main.* imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root so OPENAI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import auth, dashboard, datasets

app = FastAPI(
    title="App Reviews Analysis API",
    description="API for dashboard data: KPIs, sentiment, severity, issues, churn, trends.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(datasets.router)


@app.on_event("startup")
def on_startup() -> None:
    from api.db import init_db
    import api.models  # noqa: F401 - ensure User and Dataset are registered before create_all
    init_db()


@app.get("/")
def root() -> dict:
    return {"service": "App Reviews Analysis API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
