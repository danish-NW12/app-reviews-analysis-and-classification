"""Dashboard API routes. Analytics by dataset_id require auth and dataset ownership."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user_optional
from api.db import get_db, init_db
from api.models.dataset import Dataset, DatasetStatus
from api.models.user import User
from api.schemas.dashboard import DashboardResponse, KPIsSchema
from api.services.dashboard_service import get_dashboard_data, PROJECT_ROOT

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get(
    "/analytics",
    response_model=DashboardResponse,
    summary="Full dashboard data",
    description="Returns KPIs, charts, and tables. Use dataset_id for a completed dataset (requires auth and ownership).",
)
def get_analytics(
    dataset_id: str | None = None,
    output_dir: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> dict:
    """
    If dataset_id is provided: requires auth; returns stored analytics only if dataset belongs to current user.
    Otherwise, if output_dir is provided, run pipeline on that directory (legacy, no auth).
    If neither, run pipeline on default output/ (legacy, no auth).
    """
    if dataset_id:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required to view a dataset.")
        init_db()
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        ).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        if dataset.status != DatasetStatus.COMPLETED:
            raise HTTPException(
                status_code=503,
                detail=f"Dataset analysis is {dataset.status}. Analytics available when status is completed.",
            )
        data = dataset.analytics_json or {}
        if data.get("error"):
            raise HTTPException(status_code=503, detail=data["error"])
        return data

    out_dir = Path(output_dir) if output_dir else None
    data = get_dashboard_data(output_dir=out_dir)
    if data.get("error"):
        raise HTTPException(status_code=503, detail=data["error"])
    return data


@router.get(
    "/kpis",
    response_model=KPIsSchema,
    summary="Dashboard KPIs only",
)
def get_kpis(
    dataset_id: str | None = None,
    output_dir: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> dict:
    """Return only the KPI metrics. Use dataset_id for a completed dataset (requires auth and ownership)."""
    if dataset_id:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required.")
        init_db()
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        ).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        if dataset.status != DatasetStatus.COMPLETED:
            raise HTTPException(status_code=503, detail=f"Dataset status: {dataset.status}.")
        data = dataset.analytics_json or {}
        return data.get("kpis", {})

    out_dir = Path(output_dir) if output_dir else None
    data = get_dashboard_data(output_dir=out_dir)
    if data.get("error"):
        raise HTTPException(status_code=503, detail=data["error"])
    return data["kpis"]
