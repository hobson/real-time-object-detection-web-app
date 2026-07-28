"""One-off schema bridge for the 2026-07-26 orm.py change (added
SubmittedImage.capture_metadata and DetectionLabel.source).

This repo deliberately has no migration tooling (see curation.py's module
docstring) - `python orm.py`'s `create_all()` only creates missing tables,
it does not add columns to ones that already exist, so a plain redeploy of
orm.py would leave taco's live tables out of sync with the code and every
persist_submission() call would start silently failing (best-effort
persistence swallows the exception - see persist.py). Run this once after
deploying the new orm.py/persist.py, then it's safe to delete; it's not a
permanent migrations system, just a bridge for this one change.

Usage (from inference-server/, same venv as main.py):
    python migrate_2026_07_26_capture_metadata.py
"""
from sqlalchemy import text

from orm import engine_from_env

STATEMENTS = [
    "ALTER TABLE submitted_images ADD COLUMN IF NOT EXISTS capture_metadata JSON",
    "ALTER TABLE detection_labels ADD COLUMN IF NOT EXISTS source VARCHAR(16) NOT NULL DEFAULT 'server'",
]

if __name__ == "__main__":
    engine = engine_from_env()
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Running: {stmt}")
            conn.execute(text(stmt))
    print(f"Migration applied to {engine.url}")
