"""DB maintenance utilities for inference-server's orm.py.

Usage (from inference-server/, same venv as main.py):
    python clean_db.py [--limit N]
"""
import argparse
import logging
from pathlib import Path

from orm import Image, engine_from_env
from persist import _store_thumbnail
from sqlalchemy.orm import Session

logger = logging.getLogger("clean_db")


def abbreviate_hash(value: str, length: int = 12) -> str:
    """Shortens a sha256 (or any long hex string) for table display -
    `"a1b2c3...".` Framework-agnostic on purpose (no Flask/Markup/url_for
    here): curation.py imports this and combines it with url_for() to build
    the actual clickable column formatter, keeping the plain string
    abbreviation logic reusable/testable independent of Flask-Admin."""
    return value if len(value) <= length else f"{value[:length]}..."


def backfill_missing_thumbnails(session: Session, limit: int | None = None) -> int:
    """Finds every Image row with no thumbnail_path and generates one from
    its stored full-resolution file, reusing persist._store_thumbnail (the
    same function persist_submission() already calls for new uploads)
    rather than reimplementing the resize/save logic. Returns how many were
    backfilled.

    Best-effort per row, matching persist.py's own contract for
    thumbnailing: a missing source file or a decode failure logs a warning/
    exception and is skipped, rather than aborting the whole run."""
    query = session.query(Image).filter(Image.thumbnail_path.is_(None))
    if limit:
        query = query.limit(limit)
    images = query.all()

    backfilled = 0
    for image in images:
        path = Path(image.file_path)
        if not path.exists():
            logger.warning("Image %s: source file missing at %s - skipping", image.id, path)
            continue
        thumbnail_path = _store_thumbnail(path.read_bytes(), image.sha256)
        if thumbnail_path is None:
            continue  # _store_thumbnail already logged the failure
        image.thumbnail_path = thumbnail_path
        backfilled += 1

    session.commit()
    return backfilled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap how many images to process")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    engine = engine_from_env()
    with Session(engine) as session:
        n = backfill_missing_thumbnails(session, args.limit)
        print(f"[clean-db] Backfilled {n} thumbnail(s)")


if __name__ == "__main__":
    main()
