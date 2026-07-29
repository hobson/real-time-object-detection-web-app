"""Tags approved (training-status) images that need a human's eyes, using
the cross_model_agreement/label_quality_score compute_label_confidence.py
already computed, then reports the worst-scored images with direct
Flask-Admin edit links.

Two review-flag tag families (items 4+5):
  - "review <class> labels": an image tagged with a vehicle-bearing source
    (kitti/coco128/license_plates) that has *some* label whose
    cross_model_agreement is low - one tag per flagged class per image, so
    a curator can filter admin by exactly which class needs a second look.
  - "review license_plate labels": an image where the plate detector's own
    fresh pass (persisted by compute_label_confidence.py under a model-type
    LabelSource named yolo-v9-t-384-license-plate-end2end) found a plate
    with no matching license_plate Annotation at all - i.e. a likely
    unlabeled plate.

Usage: python training/flag_review_images.py [--agreement-threshold 0.34] [--top-n 5]
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_SERVER_DIR = REPO_ROOT / "inference-server"

load_dotenv(INFERENCE_SERVER_DIR / ".env")
sys.path.insert(0, str(INFERENCE_SERVER_DIR))

from orm import Annotation, Image, Tag, engine_from_env  # noqa: E402
from review_confusion_matrix import iou  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

LICENSE_PLATE_CLASS_ID = 80
PLATE_MODEL_NAME = "yolo-v9-t-384-license-plate-end2end"
IOU_MATCH_THRESHOLD = 0.5


def _get_or_create_tag(session: Session, name: str) -> Tag:
    tag = session.query(Tag).filter_by(name=name).one_or_none()
    if tag is None:
        tag = Tag(name=name)
        session.add(tag)
        session.flush()
    return tag


def flag_low_confidence_classes(session: Session, threshold: float) -> int:
    """Item 4: per class on a vehicle-bearing image, flag it for review if
    that class's label(s) disagree with the reference models more than
    `threshold` allows (i.e. cross_model_agreement below threshold)."""
    flagged = 0
    labels = (
        session.query(Annotation)
        .join(Annotation.image)
        .filter(Annotation.cross_model_agreement.isnot(None))
        .filter(Annotation.cross_model_agreement < threshold)
        .filter(Image.training_status == Image.TRAINING_STATUS_APPROVED)
        .all()
    )
    for label in labels:
        if not label.class_name:
            continue
        tag = _get_or_create_tag(session, f"review {label.class_name} labels")
        if tag not in label.image.tags:
            label.image.tags.append(tag)
            flagged += 1
    session.commit()
    return flagged


def flag_unlabeled_plates(session: Session) -> int:
    """Item 5: the plate detector found something, but no license_plate
    Annotation on this image matches it (IoU>=0.5) - a likely missed
    plate annotation."""
    flagged = 0
    images = (
        session.query(Image)
        .filter(Image.training_status == Image.TRAINING_STATUS_APPROVED)
        .all()
    )
    for image in images:
        plate_detections = [
            d for d in image.annotations
            if d.label_source.is_model(PLATE_MODEL_NAME)
        ]
        existing_plates = [
            d for d in image.annotations
            if d.class_id == LICENSE_PLATE_CLASS_ID
            and not (d.label_source.is_model(PLATE_MODEL_NAME))
        ]
        for det in plate_detections:
            det_box = (det.x_center - det.width / 2, det.y_center - det.height / 2, det.x_center + det.width / 2, det.y_center + det.height / 2)
            matched = any(
                iou(det_box, (e.x_center - e.width / 2, e.y_center - e.height / 2, e.x_center + e.width / 2, e.y_center + e.height / 2)) >= IOU_MATCH_THRESHOLD
                for e in existing_plates
            )
            if not matched:
                tag = _get_or_create_tag(session, "review license_plate labels")
                if tag not in image.tags:
                    image.tags.append(tag)
                    flagged += 1
                break
    session.commit()
    return flagged


def report_lowest_confidence(session: Session, top_n: int) -> None:
    admin_base = f"http://localhost:{os.environ.get('ADMIN_PORT', '5001')}{os.environ.get('ADMIN_ROOT_PATH', '')}"
    images = (
        session.query(Image)
        .filter(Image.label_quality_score.isnot(None))
        .order_by(Image.label_quality_score.asc())
        .limit(top_n)
        .all()
    )
    print(f"\n[flag-review] Top {top_n} lowest-confidence images to review:")
    for image in images:
        tags = ", ".join(t.name for t in image.tags)
        url = f"{admin_base}/image/edit/?id={image.id}"
        print(f"  id={image.id:5d}  label_quality_score={image.label_quality_score:.3f}  tags=[{tags}]  {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement-threshold", type=float, default=0.34, help="Below this cross_model_agreement, flag that label's class for review.")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    engine = engine_from_env()
    with Session(engine) as session:
        n_class_flags = flag_low_confidence_classes(session, args.agreement_threshold)
        print(f"[flag-review] Flagged {n_class_flags} new (image, class) review tag(s)")
        n_plate_flags = flag_unlabeled_plates(session)
        print(f"[flag-review] Flagged {n_plate_flags} new likely-unlabeled-plate image(s)")
        report_lowest_confidence(session, args.top_n)


if __name__ == "__main__":
    main()
