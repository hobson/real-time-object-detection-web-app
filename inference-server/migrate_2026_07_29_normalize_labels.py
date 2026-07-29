"""One-off schema migration for the 2026-07-29 orm.py normalization:
SubmittedImage/DetectionLabel -> Image/Annotation, with model_name/
label_source/dataset_id collapsed into a normalized LabelSource FK, plus
Image.training_status (the new production-vs-training-corpus signal,
replacing the old `endpoint == "training_import"` string check).

Rename-in-place, not a data copy (this DB has ~20 rows locally / ~8,300 on
taco - small enough that a full ETL would be pure ceremony). Each step below
is written to be safe to re-run (checks pg_catalog before renaming/altering,
so a partial failure can be re-run without manual cleanup), EXCEPT the final
drop of now-unused columns/tables, which is deliberately commented out - run
that by hand only after eyeballing this script's own verification output.

Does NOT touch dataset_classes/dataset_images/dataset_labels rows (they're
confirmed empty/unused) beyond dropping the tables in the same manual final
step - see the DROP statements at the bottom.

Usage (from inference-server/, same venv as main.py):
    python migrate_2026_07_29_normalize_labels.py

IMPORTANT: do not run this against taco while any background training job
(training/iterative_reweight_train.py and friends) is still writing to
submitted_images/detection_labels - confirm it has finished first
(`ssh taco "ps aux | grep iterative_reweight"`).
"""
from sqlalchemy import text

from orm import Base, engine_from_env


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        text("SELECT to_regclass(:name) IS NOT NULL"), {"name": name}
    ).scalar()


def _column_exists(conn, table: str, column: str) -> bool:
    return conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    ).scalar() is not None


def _rename_table(conn, old: str, new: str) -> None:
    if _table_exists(conn, old) and not _table_exists(conn, new):
        print(f"Renaming table {old} -> {new}")
        conn.execute(text(f"ALTER TABLE {old} RENAME TO {new}"))
    else:
        print(f"Skipping table rename {old} -> {new} (already done or {old} missing)")


def _rename_column(conn, table: str, old: str, new: str) -> None:
    if _column_exists(conn, table, old) and not _column_exists(conn, table, new):
        print(f"Renaming column {table}.{old} -> {table}.{new}")
        conn.execute(text(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}"))
    else:
        print(f"Skipping column rename {table}.{old} -> {table}.{new} (already done or missing)")


def main():
    engine = engine_from_env()
    with engine.begin() as conn:
        # 1. Rename tables in place.
        _rename_table(conn, "submitted_images", "images")
        _rename_table(conn, "detection_labels", "annotations")
        _rename_table(conn, "submitted_image_tags", "image_tags")
        _rename_column(conn, "annotations", "submitted_image_id", "image_id")
        _rename_column(conn, "image_tags", "submitted_image_id", "image_id")

        # 2. New images.training_status column + one-time backfill.
        conn.execute(text(
            "ALTER TABLE images ADD COLUMN IF NOT EXISTS training_status "
            "VARCHAR(16) NOT NULL DEFAULT 'unreviewed'"
        ))
        result = conn.execute(text(
            "UPDATE images SET training_status = 'approved' "
            "WHERE endpoint = 'training_import' AND training_status = 'unreviewed'"
        ))
        print(f"Backfilled training_status='approved' on {result.rowcount} training-imported image(s)")

    # 3. Create label_sources (entirely new table - create_all() no-ops on
    # everything that already exists post-rename above).
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        # 4. Backfill label_sources from distinct (label_source, model_name,
        # dataset_id) combos still present in annotations' old columns.
        if _column_exists(conn, "annotations", "label_source"):
            conn.execute(text("""
                INSERT INTO label_sources (source_type, dataset_id, created_at)
                SELECT DISTINCT 'dataset', dataset_id, now()
                FROM annotations
                WHERE label_source = 'human_dataset' AND dataset_id IS NOT NULL
                ON CONFLICT DO NOTHING
            """))
            conn.execute(text("""
                INSERT INTO label_sources (source_type, model_name, weights_hash, created_at)
                SELECT DISTINCT 'model', model_name, NULL, now()
                FROM annotations
                WHERE label_source = 'machine' AND model_name IS NOT NULL
                ON CONFLICT DO NOTHING
            """))

        # 5. Point every annotation at its resolved label_sources row.
        conn.execute(text(
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS label_source_id "
            "INTEGER REFERENCES label_sources(id)"
        ))
        if _column_exists(conn, "annotations", "label_source"):
            conn.execute(text("""
                UPDATE annotations a
                SET label_source_id = ls.id
                FROM label_sources ls
                WHERE a.label_source_id IS NULL
                  AND a.label_source = 'human_dataset'
                  AND ls.source_type = 'dataset'
                  AND ls.dataset_id = a.dataset_id
            """))
            conn.execute(text("""
                UPDATE annotations a
                SET label_source_id = ls.id
                FROM label_sources ls
                WHERE a.label_source_id IS NULL
                  AND a.label_source = 'machine'
                  AND ls.source_type = 'model'
                  AND ls.model_name = a.model_name
                  AND ls.weights_hash IS NULL
            """))

    # 6. Verification - print counts, do NOT drop anything automatically.
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM annotations")).scalar()
        unresolved = conn.execute(
            text("SELECT count(*) FROM annotations WHERE label_source_id IS NULL")
        ).scalar()
        print(f"annotations: {total} total, {unresolved} with no resolved label_source_id")
        if unresolved:
            print(
                "WARNING: some annotations have no label_source_id - inspect them before "
                "dropping the old model_name/label_source/dataset_id columns."
            )

    print(
        "\nMigration applied. Once you've verified the counts above, run the following "
        "BY HAND (not part of this script - irreversible):\n"
        "  ALTER TABLE annotations ALTER COLUMN label_source_id SET NOT NULL;\n"
        "  -- orm.py's Annotation.label_source_id is declared non-nullable; the column stays\n"
        "  -- nullable until this runs, so run it only once the 'unresolved' count above is 0.\n"
        "  ALTER TABLE annotations DROP COLUMN model_name, DROP COLUMN label_source, DROP COLUMN dataset_id;\n"
        "  DROP TABLE IF EXISTS dataset_labels;\n"
        "  DROP TABLE IF EXISTS dataset_images;\n"
        "  DROP TABLE IF EXISTS dataset_classes;\n"
    )


if __name__ == "__main__":
    main()
