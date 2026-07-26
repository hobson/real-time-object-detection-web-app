"""Minimal Streamlit app for hand-labeling license plates: a bounding box
(4 numbers, YOLO format) plus the plate's text and issuing state - a first
cut alternative to labelImg/CVAT/Roboflow (see docs/labeling_and_validation.md)
for when you also want to capture plate text/state, which plain YOLO boxes
have no field for.

Writes:
- <labels-dir>/<stem>.txt - one YOLO line "80 cx cy w h" per image (class 80
  = license_plate, matching data/license_plates/dataset.yaml's scheme;
  overwrites any existing box for that image - one plate per image only,
  for this first cut).
- <metadata> CSV - one row per labeled image: stem, plate_text, state.
  Kept separate from the YOLO label file since that format has no room for
  arbitrary text fields.

Usage:
    uv run --extra label streamlit run label_app.py -- \\
        --images ../data/license_plates/images \\
        --labels ../data/license_plates/labels_plates_only \\
        --metadata ../data/license_plates/plate_metadata.csv

(paths default to exactly those three, so `uv run --extra label streamlit
run label_app.py` with no args works from the `training/` directory as-is.)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_CLASS_ID = 80
BOX_COLOR = (56, 189, 248)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../data/license_plates/images")
    parser.add_argument("--labels", default="../data/license_plates/labels_plates_only")
    parser.add_argument("--metadata", default="../data/license_plates/plate_metadata.csv")
    # Streamlit forwards argv after "--" to the script; parse_known_args
    # ignores streamlit's own flags rather than erroring on them.
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
IMAGES_DIR = Path(args.images)
LABELS_DIR = Path(args.labels)
METADATA_PATH = Path(args.metadata)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data
def list_images(images_dir: str) -> list[str]:
    return sorted(
        f.name for f in Path(images_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_metadata() -> dict[str, dict[str, str]]:
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open(newline="") as f:
        return {row["stem"]: row for row in csv.DictReader(f)}


def save_metadata(metadata: dict[str, dict[str, str]]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "plate_text", "state"])
        writer.writeheader()
        for stem in sorted(metadata):
            writer.writerow(metadata[stem])


def load_box(stem: str) -> tuple[float, float, float, float] | None:
    label_path = LABELS_DIR / f"{stem}.txt"
    if not label_path.exists():
        return None
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts and int(parts[0]) == PLATE_CLASS_ID:
            return tuple(float(x) for x in parts[1:5])
    return None


def save_box(stem: str, box: tuple[float, float, float, float]) -> None:
    cx, cy, w, h = box
    (LABELS_DIR / f"{stem}.txt").write_text(
        f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
    )


st.set_page_config(page_title="Plate labeler", layout="wide")

images = list_images(str(IMAGES_DIR))
if not images:
    st.error(f"No images found in {IMAGES_DIR}")
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = 0
if "metadata" not in st.session_state:
    st.session_state.metadata = load_metadata()

st.session_state.index = max(0, min(st.session_state.index, len(images) - 1))
filename = images[st.session_state.index]
stem = Path(filename).stem
labeled_count = sum(1 for name in images if (LABELS_DIR / f"{Path(name).stem}.txt").exists())

nav_prev, nav_status, nav_next = st.columns([1, 3, 1])
with nav_prev:
    if st.button("Prev", use_container_width=True) and st.session_state.index > 0:
        st.session_state.index -= 1
        st.rerun()
with nav_status:
    st.markdown(
        f"<div style='text-align:center'>Image {st.session_state.index + 1} of "
        f"{len(images)} ({labeled_count} labeled) — {filename}</div>",
        unsafe_allow_html=True,
    )
with nav_next:
    if st.button("Next", use_container_width=True) and st.session_state.index < len(images) - 1:
        st.session_state.index += 1
        st.rerun()

image = Image.open(IMAGES_DIR / filename).convert("RGB")
width, height = image.size

existing_box = load_box(stem)
existing_meta = st.session_state.metadata.get(stem, {})

col_image, col_form = st.columns([2, 1])

with col_form:
    st.subheader("Bounding box (YOLO format, normalized 0-1)")
    cx = st.text_input("cx (center x)", value=f"{existing_box[0]:.6f}" if existing_box else "0.5", key=f"cx_{stem}")
    cy = st.text_input("cy (center y)", value=f"{existing_box[1]:.6f}" if existing_box else "0.5", key=f"cy_{stem}")
    w = st.text_input("w (width)", value=f"{existing_box[2]:.6f}" if existing_box else "0.1", key=f"w_{stem}")
    h = st.text_input("h (height)", value=f"{existing_box[3]:.6f}" if existing_box else "0.05", key=f"h_{stem}")

    st.subheader("Plate")
    plate_text = st.text_input("Plate number", value=existing_meta.get("plate_text", ""), key=f"plate_{stem}")
    state = st.text_input(
        "State abbreviation", value=existing_meta.get("state", ""), max_chars=2, key=f"state_{stem}"
    ).strip().upper()

    save_clicked = st.button("Save + Next", type="primary", use_container_width=True)

with col_image:
    try:
        box = tuple(float(v) for v in (cx, cy, w, h))
        preview = image.copy()
        bcx, bcy, bw, bh = box
        x0, y0 = (bcx - bw / 2) * width, (bcy - bh / 2) * height
        x1, y1 = (bcx + bw / 2) * width, (bcy + bh / 2) * height
        ImageDraw.Draw(preview).rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=3)
        st.image(preview, use_container_width=True)
    except ValueError:
        st.image(image, use_container_width=True)
        st.warning("Enter valid numbers for cx/cy/w/h to preview the box")

if save_clicked:
    try:
        box = tuple(float(v) for v in (cx, cy, w, h))
    except ValueError:
        st.error("cx/cy/w/h must all be numbers")
        st.stop()
    if not (0 <= box[0] <= 1 and 0 <= box[1] <= 1 and 0 < box[2] <= 1 and 0 < box[3] <= 1):
        st.error("cx/cy/w/h must be normalized 0-1, with w/h > 0")
        st.stop()

    save_box(stem, box)
    st.session_state.metadata[stem] = {"stem": stem, "plate_text": plate_text, "state": state}
    save_metadata(st.session_state.metadata)
    list_images.clear()

    if st.session_state.index < len(images) - 1:
        st.session_state.index += 1
    st.rerun()
