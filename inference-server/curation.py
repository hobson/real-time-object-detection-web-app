"""Flask-Admin data-curation app for inference-server's orm.py.

Standalone alternative to the `yolo-dataset-curator` project's own
Flask-Admin app — no shared code or schema between the two. Run separately
from the FastAPI inference server (different process, different port):

    python curation.py

Then open http://localhost:5001/. Mounted at the app's own root (not
`/admin`) so it also works unmodified behind a reverse proxy that strips a
path prefix before forwarding (e.g. `tailscale funnel --set-path=/admin`) -
the proxy's prefix supplies what would otherwise be `/admin` externally. Set
`ADMIN_ROOT_PATH=/admin` (matching whatever prefix is actually funneled to
this service) so Flask's own `url_for()` - used for the home-page redirect
and for every static asset link, including Flask-Admin's bundled CSS/JS -
generates paths with that prefix baked in, e.g. `/admin/static/compact.css`
instead of a root-absolute `/static/compact.css` the browser would resolve
against the wrong (unprefixed) URL.

Patterns used throughout
------------------------
_CompactMixin
    Injects compact.css + hotkeys.js on every list, detail, and home page
    (via url_for so paths stay correct behind ADMIN_ROOT_PATH).
_RootPathMiddleware
    Sets WSGI's SCRIPT_NAME so url_for() is prefix-aware; see ADMIN_ROOT_PATH
    above.
column_editable_list
    Click-to-edit cells in the list view via Flask-Admin's x-editable widget.
SearchableMixin / auto_sortable_columns / auto_searchable_columns
    From the sibling flask-admin-toolkit package (~/code/hobs/flask-admin-
    toolkit, installed as an editable dependency - see requirements.txt) -
    gives every view mode-aware full-text search (substring by default;
    quote for case-sensitive whole-word; `*` for regex) and makes every
    column, including JSON blobs like capture_metadata, sortable/
    searchable lexically. See that package's README for the full design
    rationale; it started as a generalized copy of claude-admin's search
    layer.
"""
import itertools
from functools import cached_property

from flask import Flask, send_from_directory, url_for
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.theme import Bootstrap4Theme
from flask_sqlalchemy import SQLAlchemy
from markupsafe import Markup

from flask_admin_toolkit import SearchableMixin, auto_searchable_columns, auto_sortable_columns

from orm import (
    Base, Dataset, DatasetClass, DatasetImage, DatasetLabel,
    DetectionLabel, SubmittedImage, Tag,
)
from persist import THUMBNAIL_DIR

import os

FLASK_SECRET = os.environ.get("FLASK_SECRET", "dev-secret-key-change-in-production")


class _RootPathMiddleware:
    """Make url_for() prefix-aware behind a proxy that strips ADMIN_ROOT_PATH
    before forwarding (see module docstring)."""

    def __init__(self, wsgi_app, root_path):
        self.wsgi_app = wsgi_app
        self.root_path = root_path

    def __call__(self, environ, start_response):
        environ["SCRIPT_NAME"] = self.root_path
        return self.wsgi_app(environ, start_response)


# ---------------------------------------------------------------------------
# Base view — compact margins + keyboard shortcuts on every page
# ---------------------------------------------------------------------------

class _CompactMixin:
    # cached_property: url_for's output is invariant for the app's lifetime
    # (same static filename + ADMIN_ROOT_PATH), so no need to recompute it
    # on every page render.
    @cached_property
    def extra_css(self):
        return [url_for("static", filename="compact.css")]

    @cached_property
    def extra_js(self):
        return [url_for("static", filename="hotkeys.js")]


class _BaseView(_CompactMixin, SearchableMixin, ModelView):
    pass


# ---------------------------------------------------------------------------
# Dataset curation views
# ---------------------------------------------------------------------------

class DatasetView(_BaseView):
    column_list = ["id", "name", "description", "created_at"]
    column_searchable_list = auto_searchable_columns(Dataset)
    column_sortable_list = auto_sortable_columns(Dataset)
    column_default_sort = ("created_at", True)
    form_columns = ["name", "description"]


class DatasetClassView(_BaseView):
    column_list = ["id", "dataset", "class_index", "name"]
    column_searchable_list = auto_searchable_columns(DatasetClass)
    column_filters = ["dataset"]
    column_sortable_list = auto_sortable_columns(DatasetClass)
    column_default_sort = ("class_index", False)
    form_columns = ["dataset", "class_index", "name"]


class DatasetImageView(_BaseView):
    column_list = ["id", "dataset", "file_path", "split", "width", "height", "created_at"]
    column_searchable_list = auto_searchable_columns(DatasetImage)
    column_filters = ["dataset", "split"]
    column_sortable_list = auto_sortable_columns(DatasetImage)
    column_default_sort = ("created_at", True)
    column_editable_list = ["split"]
    form_columns = ["dataset", "file_path", "width", "height", "split"]


class DatasetLabelView(_BaseView):
    column_list = [
        "id", "image", "class_index", "x_center", "y_center", "width", "height",
    ]
    column_filters = ["class_index"]
    column_sortable_list = auto_sortable_columns(DatasetLabel)
    column_default_sort = ("id", False)
    form_columns = ["image", "class_index", "x_center", "y_center", "width", "height"]


# ---------------------------------------------------------------------------
# Endpoint logging views
# ---------------------------------------------------------------------------

def _thumbnail_formatter(view, context, model, name):
    if not model.thumbnail_path:
        return ""
    src = url_for("thumbnail_file", filename=f"{model.sha256}.jpg")
    return Markup(f'<img src="{src}" style="max-height:64px;max-width:64px">')


def _description_formatter(view, context, model, name):
    if not model.description:
        return ""
    text = model.description
    return text if len(text) <= 80 else text[:77] + "..."


def _tags_formatter(view, context, model, name):
    return ", ".join(t.name for t in model.tags) if model.tags else ""


class SubmittedImageView(_BaseView):
    column_list = [
        "id", "thumbnail_path", "endpoint", "model_name", "sha256", "width",
        "height", "description", "tags", "inference_time_ms", "status_code",
        "client_ip", "received_at",
    ]
    column_labels = {"thumbnail_path": "Thumbnail"}
    column_formatters = {
        "thumbnail_path": _thumbnail_formatter,
        "description": _description_formatter,
        "tags": _tags_formatter,
    }
    # Inline textarea edit straight from the list view (same x-editable
    # pattern DatasetImageView uses for `split`) - a background-generated
    # caption is often close but not exactly what a curator wants, and
    # shouldn't require opening the full edit form just to tweak wording.
    column_editable_list = ["description"]
    # auto_searchable_columns already covers every String/JSON column
    # (sha256, client_ip, file_path, description, capture_metadata,
    # detection_metadata, ...) - "tags.name" is added on top since it's a
    # relationship, not a column, so the auto-deriver can't see it.
    column_searchable_list = [*auto_searchable_columns(SubmittedImage), "tags.name"]
    column_filters = ["endpoint", "model_name", "status_code", "received_at", "tags"]
    column_sortable_list = auto_sortable_columns(SubmittedImage)
    column_default_sort = ("received_at", True)
    # capture_metadata/detection_metadata are JSON blobs (see orm.py - one's
    # about the capture device/environment, the other's the YOLO output
    # summary) - too bulky for the list view, kept in the per-row form below.
    column_exclude_list = ["file_path", "content_type", "capture_metadata", "detection_metadata"]
    form_columns = [
        "sha256", "file_path", "width", "height", "content_type", "endpoint",
        "model_name", "client_ip", "inference_time_ms", "status_code",
        "description", "tags", "capture_metadata", "detection_metadata",
    ]


class TagView(_BaseView):
    column_list = ["id", "name"]
    column_searchable_list = auto_searchable_columns(Tag)
    column_sortable_list = auto_sortable_columns(Tag)
    column_default_sort = ("name", False)
    form_columns = ["name"]


class DetectionLabelView(_BaseView):
    column_list = [
        "id", "submitted_image", "class_name", "confidence", "source",
        "plate_text", "ocr_confidence", "region", "model_name",
    ]
    column_searchable_list = auto_searchable_columns(DetectionLabel)
    column_filters = ["class_name", "model_name", "region", "source"]
    column_sortable_list = auto_sortable_columns(DetectionLabel)
    column_default_sort = ("id", False)
    form_columns = [
        "submitted_image", "class_id", "class_name", "x_center", "y_center",
        "width", "height", "confidence", "model_name", "source", "plate_text",
        "ocr_confidence", "region", "region_confidence",
    ]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

# Keyed by endpoint (Flask-Admin derives this from each ModelView's `name`
# unless overridden - see create_app's admin.add_view calls) rather than
# duplicated alongside name/category/model there: the home page below reads
# name/category/model straight off the registered ModelView instances
# themselves, so only the free-text description - which has no other home -
# needs maintaining here. A view registered without an entry here just gets
# no description, rather than silently vanishing from the home page the way
# a second hand-maintained list could drift.
VIEW_DESCRIPTIONS = {
    "submittedimage": (
        "Every image POSTed to /predict or /alpr/predict (or frame over /alpr/ws) - "
        "thumbnail, capture EXIF/host metadata, YOLO detection summary, an "
        "LLM-generated accessibility description, and user tags."
    ),
    "detectionlabel": (
        "One row per detection (server- or client-computed) tied to a submitted "
        "image - class, box, confidence, and OCR fields for license-plate detections."
    ),
    "tag": 'User-curated labels (e.g. "flower", "license tag", "NYC") applied to submitted images.',
    "dataset": "A named collection of images + labels for training/fine-tuning a custom model.",
    "datasetclass": "The class index -> name mapping for one dataset (its own labels.txt/data.yaml, in effect).",
    "datasetimage": "One row per image belonging to a dataset, with its train/val/test split.",
    "datasetlabel": "One YOLO-format bounding-box label (class + normalized center/width/height) per dataset image.",
}


class _HomeView(_CompactMixin, AdminIndexView):
    """Landing page: a feature overview plus a linked, row-counted table of
    every registered admin view (grouped by category) - the Flask-Admin
    analogue of FastAPI's auto-generated /docs page for main.py's own
    endpoints."""

    def __init__(self, db, model_views, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = db
        self.model_views = model_views

    @expose("/")
    def index(self):
        sections = [
            {
                "category": category,
                "views": [
                    {
                        "endpoint": view.endpoint,
                        "name": view.name,
                        "description": VIEW_DESCRIPTIONS.get(view.endpoint, ""),
                        "count": self.db.session.query(view.model).count(),
                    }
                    for view in views
                ],
            }
            for category, views in itertools.groupby(self.model_views, key=lambda v: v.category)
        ]
        return self.render("index.html", sections=sections)


def create_app(db_url: str | None = None) -> Flask:
    from orm import engine_from_env

    app = Flask(__name__, static_folder="curation_static", static_url_path="/static")
    app.config["SECRET_KEY"] = FLASK_SECRET
    # NOT str(url) - SQLAlchemy's URL.__str__ always masks the password as
    # the literal text "***" (for safe logging), which would make this the
    # actual connection password and fail auth for any DB that isn't
    # trust/peer-authenticated.
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        db_url or engine_from_env().url.render_as_string(hide_password=False)
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    root_path = os.environ.get("ADMIN_ROOT_PATH", "")
    if root_path:
        app.wsgi_app = _RootPathMiddleware(app.wsgi_app, root_path)

    @app.route("/thumbnails/<path:filename>")
    def thumbnail_file(filename):
        return send_from_directory(THUMBNAIL_DIR, filename)

    # Tables are created once via `python orm.py` (see docs/user-manual.md
    # §3), not here - this avoids every gunicorn worker re-running DDL
    # schema checks against Postgres on every startup/restart.
    db = SQLAlchemy(app, model_class=Base)

    # Built once and handed to both admin.add_view() below and _HomeView, so
    # the home page reads each view's real name/category/model straight off
    # the same instance actually registered - no second hand-maintained list
    # to keep in sync (see VIEW_DESCRIPTIONS above). Order matters: same-
    # category views must stay contiguous for _HomeView's groupby.
    model_views = [
        SubmittedImageView(SubmittedImage, db, name="Submitted Images", category="Endpoint Traffic"),
        DetectionLabelView(DetectionLabel, db, name="Detection Labels", category="Endpoint Traffic"),
        TagView(Tag, db, name="Tags", category="Endpoint Traffic"),
        DatasetView(Dataset, db, name="Datasets", category="Dataset Curation"),
        DatasetClassView(DatasetClass, db, name="Dataset Classes", category="Dataset Curation"),
        DatasetImageView(DatasetImage, db, name="Dataset Images", category="Dataset Curation"),
        DatasetLabelView(DatasetLabel, db, name="Dataset Labels", category="Dataset Curation"),
    ]

    admin = Admin(
        app,
        name="Inference Server Curation",
        theme=Bootstrap4Theme(swatch="flatly"),
        index_view=_HomeView(db, model_views, url="/"),
    )
    for view in model_views:
        admin.add_view(view)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
