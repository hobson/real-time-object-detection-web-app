"""Bulk-load KITTI + COCO128 + data/license_plates into the production DB's
images/annotations tables (inference-server/orm.py), so the label-quality
investigation (compute_label_confidence.py) and the DB-driven training
refactor (train_from_db.py) have real rows to work against instead of flat
YOLO .txt files.

Why reuse images/annotations rather than a new table: the user's own
framing was "load into the database of submitted images" - source tags
("kitti"/"coco128"/"license_plates") distinguish bulk-imported training data
from real endpoint traffic in the same table, the same way Tag already
distinguishes anything else. Every image loaded here gets
`training_status="approved"` (see orm.py's Image.training_status docstring)
- that, not `endpoint`, is what makes it eligible for train_from_db.py's
manifest.

Per-source label provenance (see orm.py's LabelSource - resolved here via
persist.py's get_or_create_dataset_label_source/get_or_create_model_label_
source, so re-running this script against the same dataset/model reuses one
LabelSource row instead of creating a new one per image):

- KITTI: labels/{split}/*.txt is KITTI's own 8-class human annotation
  (remapped to this project's unified 81-class ids via KITTI_TO_UNIFIED,
  reused from train_resume_all_categories.py) -> dataset label_source,
  Dataset("KITTI"). labels_license_plate_only/{split}/*.txt is a SEPARATE
  directory entirely produced by label_plates_via_api.py calling this
  server's own /alpr/predict -> model label_source, model_name="fast-alpr"
  (kept apart from KITTI's own labels/ on disk specifically because it
  isn't part of KITTI's ground truth).
- COCO128: labels/train2017/*.txt mixes COCO128's own human-annotated boxes
  with license_plate boxes label_plates_via_api.py appended in place (see
  that script's docstring, use case 2). COCO's ground truth never includes
  a license_plate class, so the split is unambiguous by class id alone:
  class_id == LICENSE_PLATE_CLASS_ID -> model label_source,
  model_name="fast-alpr"; every other class id -> dataset label_source,
  Dataset("COCO128").
- data/license_plates/: labels_plates_only/*.txt is the original human-drawn
  plate box(es) (label_app.py) -> dataset label_source,
  Dataset("license_plates-manual"). labels/*.txt is pseudo_label_plates.py's
  regenerated superset (the same plate box(es) plus yolo12n.pt's
  pseudo-labeled COCO objects) - same class-id rule as COCO128 applies in
  reverse: class_id == LICENSE_PLATE_CLASS_ID is the human box, everything
  else is machine (model label_source, model_name="yolo12n.pt").

Idempotent: an image already loaded for a given source (same sha256, tagged
with that source's Tag) is skipped entirely, so re-running after a partial
failure doesn't duplicate rows.

Usage: python training/db_load_training_images.py [--sources kitti,coco128,license_plates] [--limit N]
"""
import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_SERVER_DIR = REPO_ROOT / "inference-server"
DATA_DIR = REPO_ROOT / "data"
KITTI_DIR = DATA_DIR / "external_datasets" / "kitti"
COCO128_DIR = DATA_DIR / "external_datasets" / "coco128"
PLATES_DIR = DATA_DIR / "license_plates"

# Load inference-server's .env (DATABASE_URL) explicitly before importing
# orm.py - python-dotenv's own load_dotenv() searches upward from the
# process cwd, which won't find inference-server/.env when this script is
# invoked from training/ or the repo root (a sibling directory, not an
# ancestor of either).
load_dotenv(INFERENCE_SERVER_DIR / ".env")
sys.path.insert(0, str(INFERENCE_SERVER_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from orm import Base, Dataset, Image, Tag, engine_from_env  # noqa: E402
from persist import (  # noqa: E402
    _annotation, _store_image, _store_thumbnail, get_or_create,
    get_or_create_dataset_label_source, get_or_create_model_label_source,
)
from sqlalchemy.orm import Session  # noqa: E402

from train_resume_all_categories import KITTI_TO_UNIFIED  # noqa: E402

LICENSE_PLATE_CLASS_ID = 80
NAMES: dict[int, str] = yaml.safe_load((PLATES_DIR / "dataset.yaml").read_text())["names"]


def _get_or_create_tag(session: Session, name: str) -> Tag:
    return get_or_create(session, Tag, name=name)


def _get_or_create_dataset(session: Session, name: str, description: str) -> Dataset:
    return get_or_create(session, Dataset, name=name, defaults={"description": description})


def _already_loaded(session: Session, sha256: str, source_tag: Tag) -> bool:
    return (
        session.query(Image)
        .filter(Image.sha256 == sha256, Image.tags.contains(source_tag))
        .first()
        is not None
    )


def _parse_yolo_label_file(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Returns [(class_id, (x0,y0,x1,y1) normalized), ...] - converts YOLO's
    stored center/width/height back to the (x0,y0,x1,y1) box shape
    persist._annotation's `det["box"]` expects."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.append((cid, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def _load_image_row(
    session: Session, image_path: Path, source_tag: Tag, endpoint: str
) -> Image | None:
    data = image_path.read_bytes()
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            content_type = PILImage.MIME.get(img.format, "image/jpeg")
    except Exception:
        return None

    sha256, file_path = _store_image(data, content_type)
    if _already_loaded(session, sha256, source_tag):
        return None
    thumbnail_path = _store_thumbnail(data, sha256)

    submitted = Image(
        sha256=sha256,
        file_path=file_path,
        thumbnail_path=thumbnail_path,
        width=width,
        height=height,
        content_type=content_type,
        endpoint=endpoint,
        # Curated-dataset images are trusted for training by construction -
        # see orm.py's Image.training_status docstring; this is the signal
        # train_from_db.py's manifest builder actually filters on.
        training_status=Image.TRAINING_STATUS_APPROVED,
    )
    submitted.tags.append(source_tag)
    session.add(submitted)
    session.flush()
    return submitted


def _add_labels(
    session: Session,
    submitted: Image,
    boxes: list[tuple[int, tuple]],
    *,
    human_class_ids: set[int] | None,
    dataset_label_source_id: int,
    machine_label_source_id: int,
    remap: dict[int, int | None] | None = None,
) -> None:
    """human_class_ids=None means "every box in this file is human" (KITTI's
    own labels/ dir, where the whole file - remapped via `remap` - is ground
    truth, machine plate boxes for that split live in a wholly separate
    directory instead of being mixed into this one). When set, it's the
    predicate ("is this class id the human-authored one") used for files
    that mix both provenances in place (COCO128, license_plates/labels/).

    `dataset_label_source_id`/`machine_label_source_id` are resolved once by
    the caller (one per whole load_* run, not once per image/box) since
    each is a DB round trip and the identity is constant across the run."""
    for class_id, box in boxes:
        unified_id = remap.get(class_id, class_id) if remap else class_id
        if unified_id is None:
            continue  # e.g. KITTI's "misc" - no reasonable unified match
        is_human = human_class_ids is None or unified_id in human_class_ids
        det = {"class_id": unified_id, "class_name": NAMES.get(unified_id), "box": box}
        label_source_id = dataset_label_source_id if is_human else machine_label_source_id
        session.add(_annotation(submitted.id, det, label_source_id, source="server"))


def load_kitti(session: Session, limit: int | None) -> int:
    tag = _get_or_create_tag(session, "kitti")
    dataset = _get_or_create_dataset(
        session, "KITTI", "KITTI's own 8-class human annotations (car/van/truck/pedestrian/"
        "Person_sitting/cyclist/tram/misc), remapped to this project's unified 81-class ids."
    )
    dataset_label_source_id = get_or_create_dataset_label_source(session, dataset.id).id
    fast_alpr_source_id = get_or_create_model_label_source(session, "fast-alpr").id
    count = 0
    for split in ("train", "val"):
        images_dir = KITTI_DIR / "images" / split
        image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        for image_path in image_paths:
            if limit and count >= limit:
                return count
            submitted = _load_image_row(session, image_path, tag, endpoint="training_import")
            if submitted is None:
                continue
            human_boxes = _parse_yolo_label_file(KITTI_DIR / "labels" / split / f"{image_path.stem}.txt")
            _add_labels(
                session, submitted, human_boxes, human_class_ids=None,
                dataset_label_source_id=dataset_label_source_id,
                machine_label_source_id=fast_alpr_source_id, remap=KITTI_TO_UNIFIED,
            )
            plate_boxes = _parse_yolo_label_file(
                KITTI_DIR / "labels_license_plate_only" / split / f"{image_path.stem}.txt"
            )
            for _, box in plate_boxes:
                det = {"class_id": LICENSE_PLATE_CLASS_ID, "class_name": "license_plate", "box": box}
                session.add(_annotation(submitted.id, det, fast_alpr_source_id, source="server"))
            session.commit()
            count += 1
    return count


def load_coco128(session: Session, limit: int | None) -> int:
    tag = _get_or_create_tag(session, "coco128")
    dataset = _get_or_create_dataset(
        session, "COCO128", "Ultralytics' COCO128 sample (128 images) - standard 80-class COCO "
        "human annotations, with license_plate boxes appended in place by label_plates_via_api.py."
    )
    dataset_label_source_id = get_or_create_dataset_label_source(session, dataset.id).id
    machine_label_source_id = get_or_create_model_label_source(session, "fast-alpr").id
    human_class_ids = {i for i in range(80)}  # every COCO class except license_plate itself
    count = 0
    images_dir = COCO128_DIR / "images" / "train2017"
    for image_path in sorted(images_dir.glob("*.jpg")):
        if limit and count >= limit:
            break
        submitted = _load_image_row(session, image_path, tag, endpoint="training_import")
        if submitted is None:
            continue
        boxes = _parse_yolo_label_file(COCO128_DIR / "labels" / "train2017" / f"{image_path.stem}.txt")
        _add_labels(
            session, submitted, boxes, human_class_ids=human_class_ids,
            dataset_label_source_id=dataset_label_source_id,
            machine_label_source_id=machine_label_source_id,
        )
        session.commit()
        count += 1
    return count


def load_license_plates(session: Session, limit: int | None) -> int:
    tag = _get_or_create_tag(session, "license_plates")
    dataset = _get_or_create_dataset(
        session, "license_plates-manual", "This repo's hand-labeled license-plate photo set "
        "(label_app.py) - plate box(es) are human ground truth; other-class boxes in the same "
        "files were pseudo-labeled by yolo12n.pt (see pseudo_label_plates.py)."
    )
    dataset_label_source_id = get_or_create_dataset_label_source(session, dataset.id).id
    machine_label_source_id = get_or_create_model_label_source(session, "yolo12n.pt").id
    human_class_ids = {LICENSE_PLATE_CLASS_ID}
    count = 0
    images_dir = PLATES_DIR / "images"
    for image_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        if limit and count >= limit:
            break
        submitted = _load_image_row(session, image_path, tag, endpoint="training_import")
        if submitted is None:
            continue
        boxes = _parse_yolo_label_file(PLATES_DIR / "labels" / f"{image_path.stem}.txt")
        _add_labels(
            session, submitted, boxes, human_class_ids=human_class_ids,
            dataset_label_source_id=dataset_label_source_id,
            machine_label_source_id=machine_label_source_id,
        )
        session.commit()
        count += 1
    return count


LOADERS = {"kitti": load_kitti, "coco128": load_coco128, "license_plates": load_license_plates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="kitti,coco128,license_plates")
    parser.add_argument("--limit", type=int, default=None, help="Cap images per source (for smoke-testing).")
    args = parser.parse_args()

    engine = engine_from_env()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for source in args.sources.split(","):
            loader = LOADERS[source.strip()]
            n = loader(session, args.limit)
            print(f"[db-load] {source}: loaded {n} new image(s)")


if __name__ == "__main__":
    main()
