"""Dataset service: run pipeline on uploaded CSV and persist analytics to DB."""

import logging
from pathlib import Path

from api.db import SessionLocal
from api.models.dataset import Dataset, DatasetStatus
from api.services.dashboard_service import (
    build_dashboard_payload_from_results,
    _ensure_project_in_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
CSV_FILENAME = "reviews.csv"


def get_dataset_dir(dataset_id: str) -> Path:
    """Return the directory where the dataset CSV is stored."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR / dataset_id


def run_pipeline_for_dataset(dataset_id: str) -> None:
    """
    Run the analysis pipeline on the dataset's CSV, then save analytics to DB.
    Call from a background task. Sets status to completed or failed.
    """
    _ensure_project_in_path()
    from main.analysis_pipeline import run_full_pipeline

    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return
        if dataset.status != DatasetStatus.IN_PROGRESS:
            return

        output_dir = get_dataset_dir(dataset_id)
        csv_path = output_dir / CSV_FILENAME
        if not csv_path.exists():
            dataset.status = DatasetStatus.FAILED
            dataset.error_message = "CSV file not found."
            db.commit()
            return

        try:
            df, results = run_full_pipeline(
                output_dir=output_dir,
                csv_names=[CSV_FILENAME],
            )
            payload = build_dashboard_payload_from_results(df, results)
            if payload.get("error"):
                dataset.status = DatasetStatus.FAILED
                dataset.error_message = payload["error"]
            else:
                dataset.analytics_json = payload
                dataset.status = DatasetStatus.COMPLETED
                dataset.error_message = None
        except Exception as e:
            logging.exception("Pipeline failed for dataset %s", dataset_id)
            dataset.status = DatasetStatus.FAILED
            dataset.error_message = str(e)[:2000]  # cap length
        db.commit()
    finally:
        db.close()
