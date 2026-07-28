"""Shared fixtures for training/tests/. Sets up sys.path the same way every
training/*.py script does (cross-directory import of inference-server's
orm.py) before any test module imports from it, and provides an in-memory
SQLite session for tests that need a real (but throwaway) DB rather than
mocking SQLAlemy's query API - orm.py's models are plain SQLAlchemy, so
Base.metadata.create_all() works identically against SQLite.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
INFERENCE_SERVER_DIR = REPO_ROOT / "inference-server"
TRAINING_DIR = REPO_ROOT / "training"

sys.path.insert(0, str(INFERENCE_SERVER_DIR))
sys.path.insert(0, str(TRAINING_DIR))


@pytest.fixture
def db_session():
    from orm import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
