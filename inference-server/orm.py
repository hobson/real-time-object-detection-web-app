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
    JSON, DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, String,
    UniqueConstraint, create_engine, func,
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
    # Opportunistic capture context sent alongside the image over multipart
    # (GPS, device orientation/acceleration, which camera - see
    # request_parsing.py) - a single flexible JSON blob rather than a column
    # per sensor field, since this server has no migration tooling (see
    # curation.py's module docstring) and the exact fields a client sends
    # will keep evolving. Null for the plain-raw-body request shape.
    capture_metadata: Mapped[dict | None] = mapped_column(JSON)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    detections: Mapped[list["DetectionLabel"]] = relationship(
        "DetectionLabel", back_populates="submitted_image", cascade="all, delete-orphan"
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
