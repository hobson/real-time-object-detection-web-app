"""One-off schema bridge for the 2026-07-27 orm.py change (added
SubmittedImage.description).

Same rationale as migrate_2026_07_26_capture_metadata.py - `create_all()`
only creates missing tables, not missing columns on existing ones. Run this
once after deploying the new orm.py/persist.py/describe.py, then it's safe
to delete.

Usage (from inference-server/, same venv as main.py):
    python migrate_2026_07_27_description.py
"""
from sqlalchemy import text

from orm import engine_from_env

STATEMENTS = [
    "ALTER TABLE submitted_images ADD COLUMN IF NOT EXISTS description TEXT",
]

if __name__ == "__main__":
    engine = engine_from_env()
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Running: {stmt}")
            conn.execute(text(stmt))
    print(f"Migration applied to {engine.url}")
