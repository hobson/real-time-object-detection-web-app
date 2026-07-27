"""SQLAlchemy ORM models for inference-server data curation.

Two independent table families:

Dataset curation (generic, reusable YOLO dataset schema — NOT tied to this
repo's on-disk `data/license_plates/` layout or its specific 81-class list):
    datasets, dataset_classes, dataset_images, dataset_labels

Endpoint request/label logging (records what was actually sent to and
returned by this server's own `/predict` and `/alpr/predict` /`/alpr/ws`
routes):
    submitted_images, detection_labels

This is a standalone implementation independent of the
`yolo-dataset-curator` project (which covers similar ground with its own
Flask-Admin app) — no shared code or schema compatibility is assumed between
the two.

Usage::

    from orm import Base, engine_from_env
    engine = engine_from_env()
    Base.metadata.create_all(engine)
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON, Column, DateTime, Enum as SqlEnum, Float, ForeignKey, Integer,
    String, Table, Text, UniqueConstraint, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ---------------------------------------------------------------------------
# Dataset curation — generic YOLO dataset schema
# ---------------------------------------------------------------------------

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    classes: Mapped[list["DatasetClass"]] = relationship(
        "DatasetClass", back_populates="dataset", cascade="all, delete-orphan"
    )
    images: Mapped[list["DatasetImage"]] = relationship(
        "DatasetImage", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dataset {self.id} {self.name!r}>"


class DatasetClass(Base):
    __tablename__ = "dataset_classes"
    __table_args__ = (UniqueConstraint("dataset_id", "class_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    class_index: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="classes")

    def __repr__(self) -> str:
        return f"<DatasetClass {self.id} {self.class_index}={self.name!r}>"


class DatasetImage(Base):
    __tablename__ = "dataset_images"
    __table_args__ = (UniqueConstraint("dataset_id", "file_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    split: Mapped[DatasetSplit] = mapped_column(
        SqlEnum(DatasetSplit), default=DatasetSplit.TRAIN, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="images")
    labels: Mapped[list["DatasetLabel"]] = relationship(
        "DatasetLabel", back_populates="image", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DatasetImage {self.id} {self.file_path!r}>"


class DatasetLabel(Base):
    __tablename__ = "dataset_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_image_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_images.id"), nullable=False
    )
    class_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x_center: Mapped[float] = mapped_column(Float, nullable=False)
    y_center: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    image: Mapped[DatasetImage] = relationship("DatasetImage", back_populates="labels")

    def __repr__(self) -> str:
        return f"<DatasetLabel {self.id} class={self.class_index}>"


# ---------------------------------------------------------------------------
# Endpoint request/label logging — this server's own inference traffic
# ---------------------------------------------------------------------------

# Many-to-many: one image can carry several user-submitted tags, and one
# tag (e.g. "car") applies across many images.
submitted_image_tags = Table(
    "submitted_image_tags",
    Base.metadata,
    Column("submitted_image_id", ForeignKey("submitted_images.id"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id"), primary_key=True),
)


class SubmittedImage(Base):
    """One row per image POSTed to `/predict` or `/alpr/predict`, or per
    frame received over `/alpr/ws`."""

    __tablename__ = "submitted_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Small (max 128px) JPEG rendered from the full image at capture time
    # (see persist.py's _store_image) - lets the admin list view show a
    # preview without decoding the full-resolution original per row. Null
    # for images stored before this column existed, or if thumbnailing
    # failed (e.g. Pillow couldn't decode the bytes).
    thumbnail_path: Mapped[str | None] = mapped_column(String(1024))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(64))
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128))
    client_ip: Mapped[str | None] = mapped_column(String(64))
    inference_time_ms: Mapped[float | None] = mapped_column(Float)
    status_code: Mapped[int | None] = mapped_column(Integer)
    # Capture context: any client-supplied GPS/orientation/acceleration/
    # camera-facing data sent alongside the image over multipart (see
    # request_parsing.py), merged with server-derived data added by
    # persist.py's _build_capture_metadata - EXIF pulled from the image
    # bytes (camera make/model/focal length/GPS, when present) and which
    # host ran inference. A single flexible JSON blob rather than a column
    # per field, since this server has no migration tooling (see
    # curation.py's module docstring) and the exact fields keep evolving.
    # Deliberately does NOT include detection results - see
    # detection_metadata below; the two are about different things (the
    # capture device/environment vs. the model's output) and were split
    # into separate columns so querying/indexing one doesn't drag in the
    # other.
    capture_metadata: Mapped[dict | None] = mapped_column(JSON)
    # The YOLO-side counterpart to capture_metadata: a summary of what was
    # detected (see persist.py's _detection_summary) - `{"count": int,
    # "classes": {class_name: count}}`. The full per-detection data (boxes,
    # confidence, etc.) already lives in DetectionLabel rows below; this is
    # just a cheap denormalized summary for admin display/search without a
    # join. Recomputed only at persist time and never mutated afterwards -
    # DetectionLabelView has no column_editable_list, so there's no path
    # that could let this drift from the rows it summarizes; don't add
    # DetectionLabel editing without revisiting this.
    detection_metadata: Mapped[dict | None] = mapped_column(JSON)
    # Generated after the fact by describe.py's background queue (a
    # multimodal LLM call is way too slow to run inline with /predict) - an
    # accessibility-alt-text-style caption plus keywords. Editable in the
    # admin (see curation.py) since the model's wording won't always be
    # exactly what a curator wants. Null until the background job gets to
    # it, or if that job hasn't been run at all yet (see describe.py's
    # `main()` for backfilling existing rows).
    description: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    detections: Mapped[list["DetectionLabel"]] = relationship(
        "DetectionLabel", back_populates="submitted_image", cascade="all, delete-orphan"
    )
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=submitted_image_tags, back_populates="images"
    )

    def __repr__(self) -> str:
        return f"<SubmittedImage {self.id} endpoint={self.endpoint!r} sha256={self.sha256[:8]}>"


class DetectionLabel(Base):
    """One row per detection an endpoint returned for a `SubmittedImage`,
    OR per detection a client attached to its own upload (`source`
    distinguishes the two - see request_parsing.py's `client_detections`).

    OCR fields (`plate_text`, `ocr_confidence`, `region`,
    `region_confidence`) are populated only by `/alpr/predict`/`/alpr/ws`
    (fast-alpr) detections; they're null for plain `/predict` (COCO YOLO)
    detections, which have no OCR stage.
    """

    __tablename__ = "detection_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submitted_image_id: Mapped[int] = mapped_column(
        ForeignKey("submitted_images.id"), nullable=False
    )
    class_id: Mapped[int | None] = mapped_column(Integer)
    class_name: Mapped[str | None] = mapped_column(String(64))
    x_center: Mapped[float] = mapped_column(Float, nullable=False)
    y_center: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(128))
    plate_text: Mapped[str | None] = mapped_column(String(64))
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(16))
    region_confidence: Mapped[float | None] = mapped_column(Float)
    # "server" (default) for detections this server itself computed,
    # "client" for a detection payload the caller attached to its own
    # multipart upload (e.g. a phone that already ran in-browser YOLO and
    # wants both its own and the server's results recorded side by side).
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="server", server_default="server"
    )

    submitted_image: Mapped[SubmittedImage] = relationship(
        "SubmittedImage", back_populates="detections"
    )

    def __repr__(self) -> str:
        return f"<DetectionLabel {self.id} class={self.class_name} plate={self.plate_text!r}>"


class Tag(Base):
    """A user-submitted label for a SubmittedImage (e.g. "flower", "license
    tag", "NYC") - many-to-many via submitted_image_tags, since one image
    can carry several tags and one tag applies across many images.

    Deliberately case-sensitive and space-permitting (plain String equality
    and uniqueness, no normalization/lowercasing) - a curator's own
    vocabulary (e.g. "NYC" vs "nyc", "license tag") shouldn't be silently
    mangled.
    """

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    images: Mapped[list[SubmittedImage]] = relationship(
        "SubmittedImage", secondary=submitted_image_tags, back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<Tag {self.id} {self.name!r}>"


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def engine_from_env(url: str | None = None):
    """Create a SQLAlchemy engine from the DATABASE_URL env var (loaded from
    a local `.env` via python-dotenv) or an explicit argument."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return create_engine(
        url or os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/inference_server"
        ),
        echo=False,
    )


if __name__ == "__main__":
    engine = engine_from_env()
    Base.metadata.create_all(engine)
    print(f"Tables created in {engine.url}")
    for t in Base.metadata.sorted_tables:
        print(f"  {t.name}: {[c.name for c in t.columns]}")
