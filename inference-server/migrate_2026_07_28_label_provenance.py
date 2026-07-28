"""One-off schema bridge for the 2026-07-28 orm.py changes: label provenance
and cross-model confidence scoring, added so bulk-loaded training-dataset
images (KITTI/COCO128/license_plates - see
training/db_load_training_images.py) can share the same submitted_images/
detection_labels tables as real endpoint traffic without losing track of
which labels are human ground truth vs. machine-produced, or how much
independent models agree with each label.

- DetectionLabel.label_source: "machine" (default, matches every existing
  row) or "human_dataset".
- DetectionLabel.dataset_id: FK to datasets.id, set only when
  label_source="human_dataset".
- DetectionLabel.cross_model_agreement, SubmittedImage.label_quality_score:
  both null until training/compute_label_confidence.py scores them - no
  backfill needed, existing rows simply stay unscored.

Same rationale as the other migrate_*.py scripts - create_all() only creates
missing tables, not missing columns on existing ones. Run this once after
deploying the new orm.py, then it's safe to delete.

Usage (from inference-server/, same venv as main.py):
    python migrate_2026_07_28_label_provenance.py
"""
from sqlalchemy import text

from orm import Base, engine_from_env

STATEMENTS = [
    "ALTER TABLE detection_labels ADD COLUMN IF NOT EXISTS label_source VARCHAR(16) NOT NULL DEFAULT 'machine'",
    "ALTER TABLE detection_labels ADD COLUMN IF NOT EXISTS dataset_id INTEGER REFERENCES datasets(id)",
    "ALTER TABLE detection_labels ADD COLUMN IF NOT EXISTS cross_model_agreement FLOAT",
    "ALTER TABLE submitted_images ADD COLUMN IF NOT EXISTS label_quality_score FLOAT",
]

if __name__ == "__main__":
    engine = engine_from_env()
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Running: {stmt}")
            conn.execute(text(stmt))
    # datasets/dataset_classes/dataset_images/dataset_labels already exist in
    # every deployed DB (created by an earlier migration) - this is a no-op
    # safety net, not the actual point of this script.
    Base.metadata.create_all(engine)
    print(f"Migration applied to {engine.url}")
