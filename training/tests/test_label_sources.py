"""persist.py's get_or_create_*_label_source dedup behavior - the whole
point of normalizing model_name/label_source/dataset_id into LabelSource is
that re-scoring the same checkpoint or re-importing the same dataset reuses
one row rather than creating a new identity every run."""
from orm import Dataset, LabelSource
from persist import get_or_create_dataset_label_source, get_or_create_model_label_source


def test_model_label_source_same_model_and_hash_reuses_row(db_session):
    first = get_or_create_model_label_source(db_session, "yolo12n", weights_hash="abc123")
    second = get_or_create_model_label_source(db_session, "yolo12n", weights_hash="abc123")
    assert first.id == second.id
    assert db_session.query(LabelSource).filter_by(source_type="model", model_name="yolo12n").count() == 1


def test_model_label_source_different_weights_hash_creates_new_row(db_session):
    """Simulates re-training producing a new checkpoint under the same
    model_name (e.g. a new training round) - a different weights_hash must
    be tracked as a genuinely different identity, not silently merged."""
    first = get_or_create_model_label_source(db_session, "license_plate_ft", weights_hash="round1hash")
    second = get_or_create_model_label_source(db_session, "license_plate_ft", weights_hash="round2hash")
    assert first.id != second.id


def test_model_label_source_null_weights_hash_reused_within_a_session(db_session):
    """Production traffic (persist.py) resolves label sources with
    weights_hash=None - get-or-create still reuses one row across calls
    within the same querying session (SQLAlchemy's IS NULL matching), even
    though the DB-level partial unique index treats separate NULL rows as
    distinct (see orm.py's LabelSource docstring) - that wrinkle only
    matters for concurrent/historical rows, not this normal path."""
    first = get_or_create_model_label_source(db_session, "yolo12n.onnx")
    second = get_or_create_model_label_source(db_session, "yolo12n.onnx")
    assert first.id == second.id


def test_model_label_source_unknown_model_name_when_none(db_session):
    label_source = get_or_create_model_label_source(db_session, None)
    assert label_source.model_name == "unknown"


def test_dataset_label_source_dedup_by_dataset_id(db_session):
    kitti = Dataset(name="KITTI")
    coco = Dataset(name="COCO128")
    db_session.add_all([kitti, coco])
    db_session.flush()

    first = get_or_create_dataset_label_source(db_session, kitti.id)
    second = get_or_create_dataset_label_source(db_session, kitti.id)
    third = get_or_create_dataset_label_source(db_session, coco.id)

    assert first.id == second.id
    assert first.id != third.id
