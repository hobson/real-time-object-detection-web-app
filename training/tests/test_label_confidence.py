"""compute_label_confidence.py's per-label/per-image aggregation math,
against a real (in-memory SQLite) DB rather than mocks - both functions
under test do real ORM queries/writes, which mocking would just re-assert
rather than verify."""
from compute_label_confidence import _persist_new_detections, _score_image
from orm import DetectionLabel, SubmittedImage


def _make_image(session) -> SubmittedImage:
    image = SubmittedImage(sha256="a" * 64, file_path="/tmp/x.jpg", endpoint="training_import")
    session.add(image)
    session.flush()
    return image


def _make_label(session, image, class_id, box, model_name=None) -> DetectionLabel:
    x0, y0, x1, y1 = box
    label = DetectionLabel(
        submitted_image_id=image.id, class_id=class_id, class_name=str(class_id),
        x_center=(x0 + x1) / 2, y_center=(y0 + y1) / 2, width=x1 - x0, height=y1 - y0,
        model_name=model_name, label_source="human_dataset" if model_name is None else "machine",
    )
    session.add(label)
    session.flush()
    return label


def test_persist_new_detections_skips_matching_existing_row(db_session):
    image = _make_image(db_session)
    _make_label(db_session, image, class_id=2, box=(0.1, 0.1, 0.5, 0.5), model_name="modelA")
    added = _persist_new_detections(db_session, image, "modelA", [(2, (0.1, 0.1, 0.5, 0.5))])
    assert added == 0
    assert db_session.query(DetectionLabel).filter_by(submitted_image_id=image.id).count() == 1


def test_persist_new_detections_adds_when_no_matching_row(db_session):
    image = _make_image(db_session)
    added = _persist_new_detections(db_session, image, "modelA", [(2, (0.1, 0.1, 0.5, 0.5))])
    assert added == 1
    rows = db_session.query(DetectionLabel).filter_by(submitted_image_id=image.id).all()
    assert len(rows) == 1
    assert rows[0].model_name == "modelA"
    assert rows[0].label_source == "machine"


def test_persist_new_detections_different_class_not_deduped(db_session):
    image = _make_image(db_session)
    _make_label(db_session, image, class_id=2, box=(0.1, 0.1, 0.5, 0.5), model_name="modelA")
    # Same box, different class - not a match, should be added as a second row.
    added = _persist_new_detections(db_session, image, "modelA", [(7, (0.1, 0.1, 0.5, 0.5))])
    assert added == 1


def test_score_image_agreement_fraction(db_session):
    image = _make_image(db_session)
    label = _make_label(db_session, image, class_id=2, box=(0.0, 0.0, 1.0, 1.0))
    # 2 of 3 reference models confirm this label (same class, matching box);
    # the third finds nothing relevant.
    models_and_detections = [
        ("model1", [(2, (0.0, 0.0, 1.0, 1.0))]),
        ("model2", [(2, (0.0, 0.0, 1.0, 1.0))]),
        ("model3", [(9, (0.0, 0.0, 1.0, 1.0))]),  # wrong class - doesn't count
    ]
    # No refresh() here: _score_image() sets these attributes in memory and
    # relies on its caller (main()'s per-image loop) to commit afterward -
    # refresh() would reload from the DB *without* an intervening flush of
    # these new values and wipe them back to their pre-call state, which
    # would be testing an artifact of this test's own ordering, not the
    # function.
    _score_image(db_session, image, models_and_detections)
    assert label.cross_model_agreement == 2 / 3
    assert image.label_quality_score == 2 / 3


def test_score_image_no_labels_leaves_quality_score_none(db_session):
    image = _make_image(db_session)
    _score_image(db_session, image, [("model1", [])])
    assert image.label_quality_score is None


def test_score_image_averages_across_multiple_labels(db_session):
    image = _make_image(db_session)
    _make_label(db_session, image, class_id=2, box=(0.0, 0.0, 0.5, 0.5))  # will be confirmed
    _make_label(db_session, image, class_id=9, box=(0.5, 0.5, 1.0, 1.0))  # will not be confirmed
    models_and_detections = [("model1", [(2, (0.0, 0.0, 0.5, 0.5))])]
    _score_image(db_session, image, models_and_detections)
    # One label scores 1/1, the other 0/1 - mean is 0.5.
    assert image.label_quality_score == 0.5
