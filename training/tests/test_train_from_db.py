"""train_from_db.py's build_db_manifest training_status filter - proves the
manifest is keyed on Image.training_status, not on `endpoint`, so a
curator-promoted production image (endpoint='predict', training_status=
'approved') is included exactly like a bulk-imported one.

Writes real symlinks/files via build_db_manifest (matching what it does in
production), so GENERATED_DIR/GENERATED_TRAIN_MANIFEST are monkeypatched to
a pytest tmp_path rather than the real data/external_datasets/db_export/ -
that directory is shared with the actual training pipeline and must not be
polluted by test runs.
"""
import train_from_db
from orm import Annotation, Dataset, Image, LabelSource


def _make_image(session, *, endpoint, training_status) -> Image:
    image = Image(
        sha256="a" * 64, file_path="/tmp/x.jpg", endpoint=endpoint, training_status=training_status,
    )
    session.add(image)
    session.flush()
    return image


def _make_dataset_label_source(session) -> LabelSource:
    dataset = Dataset(name="test-dataset")
    session.add(dataset)
    session.flush()
    label_source = LabelSource(source_type="dataset", dataset_id=dataset.id)
    session.add(label_source)
    session.flush()
    return label_source


def _add_trusted_label(session, image, label_source) -> Annotation:
    label = Annotation(
        image_id=image.id, label_source_id=label_source.id, class_id=2, class_name="car",
        x_center=0.5, y_center=0.5, width=0.2, height=0.2,
    )
    session.add(label)
    session.flush()
    return label


def _patch_generated_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(train_from_db, "GENERATED_DIR", tmp_path)
    monkeypatch.setattr(train_from_db, "GENERATED_TRAIN_MANIFEST", tmp_path / "train.generated.txt")


def test_build_db_manifest_includes_only_approved_images(db_session, tmp_path, monkeypatch):
    _patch_generated_paths(monkeypatch, tmp_path)
    label_source = _make_dataset_label_source(db_session)
    unreviewed = _make_image(db_session, endpoint="predict", training_status="unreviewed")
    approved = _make_image(db_session, endpoint="training_import", training_status="approved")
    rejected = _make_image(db_session, endpoint="predict", training_status="rejected")
    for image in (unreviewed, approved, rejected):
        _add_trusted_label(db_session, image, label_source)

    n_images, _, _ = train_from_db.build_db_manifest(
        db_session, current_model_name="none", agreement_threshold=0.67, oversample_repeats=3, fraction=1.0,
    )

    assert n_images == 1
    manifest_text = (tmp_path / "train.generated.txt").read_text()
    assert f"/{approved.id}." in manifest_text
    assert f"/{unreviewed.id}." not in manifest_text
    assert f"/{rejected.id}." not in manifest_text


def test_build_db_manifest_ignores_endpoint_field(db_session, tmp_path, monkeypatch):
    """A curator-promoted production capture (endpoint='predict',
    training_status='approved') must be included - proves the filter
    genuinely switched from endpoint to training_status rather than just
    relabeling the same signal."""
    _patch_generated_paths(monkeypatch, tmp_path)
    label_source = _make_dataset_label_source(db_session)
    promoted = _make_image(db_session, endpoint="predict", training_status="approved")
    _add_trusted_label(db_session, promoted, label_source)

    n_images, _, _ = train_from_db.build_db_manifest(
        db_session, current_model_name="none", agreement_threshold=0.67, oversample_repeats=3, fraction=1.0,
    )

    assert n_images == 1


class _FakeTrainer:
    """Duck-typed stand-in for ultralytics' BaseTrainer - _save_best_by_f1
    only reads trainer.metrics/.epoch and reads+writes trainer.last/.best
    (real Path objects, backed by tmp_path so no ultralytics/torch
    checkpoint format is needed - the callback only copies bytes, it never
    parses them)."""

    def __init__(self, tmp_path, precision, recall, epoch=0):
        self.metrics = {"metrics/precision(B)": precision, "metrics/recall(B)": recall}
        self.epoch = epoch
        self.last = tmp_path / "last.pt"
        self.best = tmp_path / "best.pt"


def test_save_best_by_f1_saves_on_first_validation(tmp_path):
    trainer = _FakeTrainer(tmp_path, precision=0.8, recall=0.6)
    trainer.last.write_bytes(b"epoch0-weights")

    train_from_db._save_best_by_f1(trainer)

    assert trainer.best.read_bytes() == b"epoch0-weights"
    assert trainer._best_f1 == 2 * 0.8 * 0.6 / (0.8 + 0.6)


def test_save_best_by_f1_skips_when_f1_does_not_improve(tmp_path):
    trainer = _FakeTrainer(tmp_path, precision=0.8, recall=0.6)
    trainer.last.write_bytes(b"epoch0-weights")
    train_from_db._save_best_by_f1(trainer)

    # Next epoch: worse F1 (lower precision, same recall) - best.pt must NOT
    # be overwritten with this epoch's (worse) weights.
    trainer.metrics = {"metrics/precision(B)": 0.5, "metrics/recall(B)": 0.6}
    trainer.epoch = 1
    trainer.last.write_bytes(b"epoch1-weights")
    train_from_db._save_best_by_f1(trainer)

    assert trainer.best.read_bytes() == b"epoch0-weights"


def test_save_best_by_f1_overwrites_when_f1_improves(tmp_path):
    trainer = _FakeTrainer(tmp_path, precision=0.8, recall=0.6)
    trainer.last.write_bytes(b"epoch0-weights")
    train_from_db._save_best_by_f1(trainer)

    trainer.metrics = {"metrics/precision(B)": 0.9, "metrics/recall(B)": 0.9}
    trainer.epoch = 1
    trainer.last.write_bytes(b"epoch1-weights")
    train_from_db._save_best_by_f1(trainer)

    assert trainer.best.read_bytes() == b"epoch1-weights"
    assert trainer._best_f1 == 0.9


def test_save_best_by_f1_handles_missing_metrics(tmp_path):
    """A skipped-validation epoch (e.g. val runs only every N epochs) must
    not crash or touch best.pt."""
    trainer = _FakeTrainer(tmp_path, precision=None, recall=None)
    trainer.last.write_bytes(b"epoch0-weights")

    train_from_db._save_best_by_f1(trainer)

    assert not trainer.best.exists()
