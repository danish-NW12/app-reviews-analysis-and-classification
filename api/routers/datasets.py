"""Dataset management API: upload, list, get, delete. All require authentication."""

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from api.auth import CurrentUser
from api.db import get_db, init_db
from api.models.dataset import Dataset, DatasetStatus
from api.services.dataset_service import get_dataset_dir, run_pipeline_for_dataset, CSV_FILENAME

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Expected CSV columns (same structure as existing data)
REQUIRED_CSV_COLUMNS = {"content", "score"}


def _validate_csv_columns(content: bytes) -> None:
    """Raise if first line does not contain required columns."""
    first_line = content.split(b"\n")[0].decode("utf-8", errors="replace").strip()
    if not first_line:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    headers = {h.strip().lower() for h in first_line.split(",")}
    if not REQUIRED_CSV_COLUMNS.issubset(headers):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must include columns: {sorted(REQUIRED_CSV_COLUMNS)}. Found: {sorted(headers)}.",
        )


@router.post("", status_code=201)
def create_dataset(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    name: str = Form(..., min_length=1, max_length=255),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Create a new dataset: upload a CSV and start analysis.
    Returns the created dataset (status in_progress). When analysis completes, status becomes completed.
    """
    init_db()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    content = file.file.read()
    _validate_csv_columns(content)

    dataset_id = str(uuid4())
    output_dir = get_dataset_dir(dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    try:
        with open(csv_path, "wb") as f:
            f.write(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    dataset = Dataset(
        id=dataset_id,
        user_id=current_user.id,
        name=name.strip(),
        filename=file.filename or "reviews.csv",
        file_path=str(csv_path),
        status=DatasetStatus.IN_PROGRESS,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    background_tasks.add_task(run_pipeline_for_dataset, dataset_id)

    return dataset.to_dict()


@router.get("")
def list_datasets(current_user: CurrentUser, db: Session = Depends(get_db)):
    """List current user's datasets with name, filename, status."""
    init_db()
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return [d.to_dict() for d in datasets]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, current_user: CurrentUser, db: Session = Depends(get_db)):
    """Get one dataset by id (must belong to current user)."""
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset.to_dict()


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, current_user: CurrentUser, db: Session = Depends(get_db)):
    """Delete a dataset and its stored file (must belong to current user)."""
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    output_dir = get_dataset_dir(dataset_id)
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass

    db.delete(dataset)
    db.commit()
    return None
