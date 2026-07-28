"""One-off schema bridge for the 2026-07-27 orm.py changes:

- split SubmittedImage.capture_metadata into two columns - capture_metadata
  keeps the capture-device/environment data (EXIF, client GPS/orientation,
  server host), and the new detection_metadata column takes over the
  YOLO-detection-summary data that used to be merged into the same blob.
- added Tag + the submitted_image_tags association table (brand new
  tables, so `Base.metadata.create_all()` would handle these on its own -
  they're included here too just so one script brings a DB fully up to
  date after this change).

Same rationale as migrate_2026_07_26_capture_metadata.py - `create_all()`
only creates missing tables, not missing columns on existing ones. Run this
once after deploying the new orm.py/persist.py, then it's safe to delete.

Usage (from inference-server/, same venv as main.py):
    python migrate_2026_07_27_detection_metadata_tags.py
"""
from sqlalchemy import text
from sqlalchemy.orm import Session

from orm import Base, SubmittedImage, engine_from_env

STATEMENTS = [
    "ALTER TABLE submitted_images ADD COLUMN IF NOT EXISTS detection_metadata JSON",
]


def _move_existing_detection_summaries(engine) -> None:
    """Data migration, not just schema: rows written before this split have
    their detection summary nested under capture_metadata["detection_
    summary"] - move it over to the new column so old rows aren't left
    without one."""
    with Session(engine) as session:
        rows = (
            session.query(SubmittedImage)
            .filter(SubmittedImage.capture_metadata.isnot(None))
            .all()
        )
        moved = 0
        for row in rows:
            # Reassign (don't mutate in place) - SQLAlchemy's plain JSON
            # type only detects attribute *assignment*, not in-place dict
            # mutation, so popping from row.capture_metadata directly would
            # silently fail to persist.
            metadata = dict(row.capture_metadata or {})
            summary = metadata.pop("detection_summary", None)
            if summary is not None:
                row.detection_metadata = summary
                row.capture_metadata = metadata
                moved += 1
        session.commit()
    print(f"Moved detection_summary out of capture_metadata for {moved} row(s)")


if __name__ == "__main__":
    engine = engine_from_env()
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Running: {stmt}")
            conn.execute(text(stmt))
    # New tables only (tags, submitted_image_tags) - create_all() never
    # touches a table that already exists, so this is safe to run alongside
    # the ALTER above.
    Base.metadata.create_all(engine)
    _move_existing_detection_summaries(engine)
    print(f"Migration applied to {engine.url}")
