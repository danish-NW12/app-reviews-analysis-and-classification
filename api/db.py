"""Database configuration and session management."""

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'app_reviews.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and run any pending migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate_add_user_id_to_datasets()


def _migrate_add_user_id_to_datasets() -> None:
    """Add user_id column to datasets if it doesn't exist (for existing DBs)."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(datasets)"))
        rows = result.fetchall()
        if not rows:
            return  # table doesn't exist yet (fresh install)
        columns = [row[1] for row in rows]
        if "user_id" not in columns:
            conn.execute(text("ALTER TABLE datasets ADD COLUMN user_id VARCHAR(36)"))
            conn.commit()
