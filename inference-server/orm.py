"""SQLAlchemy ORM models for inference-server data curation.

Normalized around three concerns (see the DB-normalization plan this session
worked from for the full rationale):

  images          — pure image data, one row per unique upload/import,
                    regardless of why it exists (production traffic vs.
                    bulk-imported training corpus). `training_status`
                    ("unreviewed"/"approved"/"rejected") is the one field
                    that distinguishes "safe to train on" from "arrived via
                    a live endpoint, not yet reviewed" — a curator can flip
                    a production image to "approved" via Flask-Admin to
                    promote it into the training corpus.
  label_sources   — normalized, deduplicated identity of whatever produced
                    a set of annotations: a specific model checkpoint
                    (name + weights hash + major.minor version), a curated
                    dataset (KITTI/COCO128/etc — see `datasets`), or a
                    human curator editing/creating a label by hand in
                    Flask-Admin. Replaces the old per-row `model_name`
                    string + `label_source` string + `dataset_id` combo
                    that used to live directly on every annotation row.
  annotations     — the actual per-image detection/label content (box,
                    class, confidence, OCR fields), FK'd to both `images`
                    and `label_sources`. This is the images<->label_sources
                    many-to-many, made concrete as the join-with-attributes
                    table (bounding-box content is never actually shared
                    across images, so there's no value in a separate
                    content-only `labels` table plus a second join).

`datasets` is kept as a small side table naming curated datasets
(KITTI/COCO128/...) — purely a `label_sources.dataset_id` FK target now;
the earlier `DatasetClass`/`DatasetImage`/`DatasetLabel` tables that used to
sit alongside it were a parallel, never-actually-wired-up generic YOLO
dataset schema (confirmed zero rows, zero real consumers) and have been
dropped entirely in favor of the schema above.

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

from sqlalchemy import (
    JSON, Column, DateTime, Float, ForeignKey, Index, Integer,
    String, Table, Text, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Datasets — named collections a label_source can point at
# ---------------------------------------------------------------------------

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Dataset {self.id} {self.name!r}>"


# ---------------------------------------------------------------------------
# label_sources — normalized identity of whatever produced a set of labels
# ---------------------------------------------------------------------------

class LabelSource(Base):
    """One row per distinct label-producing identity:

      source_type="model"          — a specific model checkpoint, identified
                                      by model_name + weights_hash (a short
                                      sha256 prefix of the checkpoint file's
                                      bytes - see training/model_identity.py)
                                      + major_version.minor_version. major is
                                      bumped only for architecture/
                                      preprocessing changes; minor per
                                      training run. weights_hash is null for
                                      historical rows whose checkpoint file
                                      has since been overwritten on disk by a
                                      later round under the same path - it
                                      can't be retroactively recomputed, and
                                      such rows are deliberately NOT deduped
                                      against each other (see the partial
                                      unique index below).
      source_type="dataset"        — a curated dataset (KITTI/COCO128/...),
                                      dataset_id set, ground truth.
      source_type="human_curator"  — a label manually created/edited in
                                      Flask-Admin. curator_username is free
                                      text for now (curation.py has no auth
                                      system yet) - becomes a real FK to a
                                      users table once one exists.

    Deduplicated via partial unique indexes (see __table_args__) so
    re-scoring the same checkpoint, or re-importing the same dataset, reuses
    the same row instead of creating a duplicate identity per run.
    """

    __tablename__ = "label_sources"
    __table_args__ = (
        Index(
            "uq_label_sources_model",
            "model_name", "weights_hash", "major_version", "minor_version",
            unique=True,
            postgresql_where=Column("source_type") == "model",
            sqlite_where=Column("source_type") == "model",
        ),
        Index(
            "uq_label_sources_dataset",
            "dataset_id",
            unique=True,
            postgresql_where=Column("source_type") == "dataset",
            sqlite_where=Column("source_type") == "dataset",
        ),
        Index(
            "uq_label_sources_curator",
            "curator_username",
            unique=True,
            postgresql_where=Column("source_type") == "human_curator",
            sqlite_where=Column("source_type") == "human_curator",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128))
    weights_hash: Mapped[str | None] = mapped_column(String(16))
    major_version: Mapped[int | None] = mapped_column(Integer)
    minor_version: Mapped[int | None] = mapped_column(Integer)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"))
    curator_username: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset | None] = relationship("Dataset")

    @property
    def is_dataset(self) -> bool:
        return self.source_type == "dataset"

    def is_model(self, model_name: str) -> bool:
        return self.source_type == "model" and self.model_name == model_name

    def __repr__(self) -> str:
        if self.source_type == "model":
            return f"<LabelSource {self.id} model={self.model_name}@{self.weights_hash}>"
        if self.source_type == "dataset":
            return f"<LabelSource {self.id} dataset_id={self.dataset_id}>"
        return f"<LabelSource {self.id} curator={self.curator_username!r}>"


# ---------------------------------------------------------------------------
# images / annotations — every image this server has ever seen, and every
# label attached to one, regardless of production traffic vs. training data
# ---------------------------------------------------------------------------

# Many-to-many: one image can carry several user-submitted tags, and one
# tag (e.g. "car") applies across many images.
image_tags = Table(
    "image_tags",
    Base.metadata,
    Column("image_id", ForeignKey("images.id"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id"), primary_key=True),
)


class Image(Base):
    """One row per unique image this server has ever handled - a frame
    POSTed to `/predict`/`/alpr/predict`/`/alpr/ws`, or an image bulk-loaded
    from a curated training dataset (see training/db_load_training_
    images.py). `training_status` is what actually distinguishes the two
    for training purposes, not `endpoint`:

      "unreviewed" (default) — production traffic; not yet vetted for
                                training use.
      "approved"              — bulk-imported curated-dataset images, or a
                                production image a curator has manually
                                reviewed and promoted via Flask-Admin.
      "rejected"               — a curator has explicitly excluded it.
    """

    __tablename__ = "images"

    # String literals for training_status - centralized here so every query
    # that filters on it (train_from_db.py, compute_label_confidence.py,
    # flag_review_images.py) references one constant instead of repeating
    # the literal "approved".
    TRAINING_STATUS_UNREVIEWED = "unreviewed"
    TRAINING_STATUS_APPROVED = "approved"
    TRAINING_STATUS_REJECTED = "rejected"

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
    # Which route produced this row ("predict", "alpr_predict", "alpr_ws",
    # "training_import", ...) - a request-time fact kept for admin
    # filtering/search, but no longer the training-eligibility signal (see
    # training_status below for that).
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    # Which model served THIS live request - a fact about the request,
    # distinct from any one annotation's producing label_source (a request
    # can run more than one model, e.g. ALPR's detector+OCR; an image can
    # also later be re-scored by other models entirely, each getting its own
    # label_sources row without touching this column).
    model_name: Mapped[str | None] = mapped_column(String(128))
    client_ip: Mapped[str | None] = mapped_column(String(64))
    inference_time_ms: Mapped[float | None] = mapped_column(Float)
    status_code: Mapped[int | None] = mapped_column(Integer)
    # Whether this image is safe to train on - see class docstring.
    training_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unreviewed", server_default="unreviewed"
    )
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
    # confidence, etc.) already lives in Annotation rows below; this is
    # just a cheap denormalized summary for admin display/search without a
    # join. Recomputed only at persist time and never mutated afterwards -
    # AnnotationView has no column_editable_list, so there's no path
    # that could let this drift from the rows it summarizes; don't add
    # Annotation editing without revisiting this.
    detection_metadata: Mapped[dict | None] = mapped_column(JSON)
    # Generated after the fact by describe.py's background queue (a
    # multimodal LLM call is way too slow to run inline with /predict) - an
    # accessibility-alt-text-style caption plus keywords. Editable in the
    # admin (see curation.py) since the model's wording won't always be
    # exactly what a curator wants. Null until the background job gets to
    # it, or if that job hasn't been run at all yet (see describe.py's
    # `main()` for backfilling existing rows).
    description: Mapped[str | None] = mapped_column(Text)
    # Aggregate of this image's Annotation rows' cross_model_agreement
    # (see that column's docstring) - mean agreement across labels that have
    # a score, or null if this image has no scored labels yet (never 0 -
    # "not yet scored" and "scored as untrustworthy" must stay distinguishable,
    # since the former shouldn't surface in a "worst labels" review queue).
    label_quality_score: Mapped[float | None] = mapped_column(Float)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="image", cascade="all, delete-orphan"
    )
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=image_tags, back_populates="images"
    )

    def __repr__(self) -> str:
        return f"<Image {self.id} endpoint={self.endpoint!r} sha256={self.sha256[:8]}>"


class Annotation(Base):
    """One row per detection/label attached to an `Image` - the
    images<->label_sources many-to-many, made concrete: `image_id` says
    which image, `label_source_id` says who/what produced this specific
    box (a model checkpoint, a curated dataset's ground truth, or a human
    curator - see `LabelSource`).

    OCR fields (`plate_text`, `ocr_confidence`, `region`,
    `region_confidence`) are populated only by `/alpr/predict`/`/alpr/ws`
    (fast-alpr) detections; they're null for plain `/predict` (COCO YOLO)
    detections, which have no OCR stage.
    """

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False)
    label_source_id: Mapped[int] = mapped_column(ForeignKey("label_sources.id"), nullable=False)
    class_id: Mapped[int | None] = mapped_column(Integer)
    class_name: Mapped[str | None] = mapped_column(String(64))
    x_center: Mapped[float] = mapped_column(Float, nullable=False)
    y_center: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    plate_text: Mapped[str | None] = mapped_column(String(64))
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(16))
    region_confidence: Mapped[float | None] = mapped_column(Float)
    # "server" (default) for detections this server itself computed,
    # "client" for a detection payload the caller attached to its own
    # multipart upload (e.g. a phone that already ran in-browser YOLO and
    # wants both its own and the server's results recorded side by side).
    # Kept as its own column rather than folded into label_source_id: this
    # answers "who computed this at request time," orthogonal to "how did
    # this label come to exist" (label_source_id) - a client-reported
    # detection is still produced by some model (the client's own on-device
    # YOLO), which could get its own real LabelSource row if that ever
    # becomes worth distinguishing per-client-model; collapsing "server" vs
    # "client" into label_source_id today would discard that distinction by
    # merging every client's report into one identity regardless of model.
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="server", server_default="server"
    )
    # Fraction of independent reference models (see
    # training/compute_label_confidence.py) whose own fresh detection over
    # this image confirms this label (IoU>=0.5 + matching class). Null until
    # that script has scored this row; 0.0 is a real "no model agrees with
    # this label" result, so the two must stay distinguishable.
    cross_model_agreement: Mapped[float | None] = mapped_column(Float)

    image: Mapped[Image] = relationship("Image", back_populates="annotations")
    label_source: Mapped[LabelSource] = relationship("LabelSource")

    def is_trusted(self, agreement_threshold: float) -> bool:
        """Ground truth from a curated dataset is trusted unconditionally;
        a machine label is trusted only if independent reference models
        corroborate it at least `agreement_threshold` (see
        training/compute_label_confidence.py's cross_model_agreement)."""
        return self.label_source.is_dataset or (
            self.cross_model_agreement is not None and self.cross_model_agreement >= agreement_threshold
        )

    def __repr__(self) -> str:
        return f"<Annotation {self.id} class={self.class_name} plate={self.plate_text!r}>"


class Tag(Base):
    """A user-submitted label for an Image (e.g. "flower", "license tag",
    "NYC") - many-to-many via image_tags, since one image can carry several
    tags and one tag applies across many images.

    Deliberately case-sensitive and space-permitting (plain String equality
    and uniqueness, no normalization/lowercasing) - a curator's own
    vocabulary (e.g. "NYC" vs "nyc", "license tag") shouldn't be silently
    mangled.
    """

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    images: Mapped[list[Image]] = relationship(
        "Image", secondary=image_tags, back_populates="tags"
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
