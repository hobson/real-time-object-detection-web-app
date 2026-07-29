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
import os
from functools import cached_property

from clean_db import abbreviate_hash
from flask import Flask, abort, render_template, send_file, send_from_directory, url_for
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.theme import Bootstrap4Theme
from flask_admin_toolkit import SearchableMixin, auto_searchable_columns, auto_sortable_columns
from flask_sqlalchemy import SQLAlchemy
from markupsafe import Markup
from orm import Annotation, Base, Dataset, Image, LabelSource, Tag
from persist import STORAGE_DIR, THUMBNAIL_DIR

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


class LabelSourceView(_BaseView):
    column_list = [
        "id", "source_type", "model_name", "weights_hash", "major_version",
        "minor_version", "dataset", "curator_username", "created_at",
    ]
    column_searchable_list = auto_searchable_columns(LabelSource)
    column_filters = ["source_type", "model_name", "dataset", "curator_username"]
    column_sortable_list = auto_sortable_columns(LabelSource)
    column_default_sort = ("created_at", True)
    form_columns = [
        "source_type", "model_name", "weights_hash", "major_version",
        "minor_version", "dataset", "curator_username",
    ]


# ---------------------------------------------------------------------------
# Endpoint logging views
# ---------------------------------------------------------------------------

def _thumbnail_formatter(view, context, model, name):
    if not model.thumbnail_path:
        return ""
    src = url_for("thumbnail_file", filename=f"{model.sha256}.jpg")
    href = url_for("image_view_page", image_id=model.id)
    return Markup(f'<a href="{href}"><img src="{src}" style="max-height:64px;max-width:64px"></a>')


def _description_formatter(view, context, model, name):
    if not model.description:
        return ""
    text = model.description
    return text if len(text) <= 80 else text[:77] + "..."


def _tags_formatter(view, context, model, name):
    return ", ".join(t.name for t in model.tags) if model.tags else ""


<<<<<<< HEAD
=======
def _label_source_display_name(label_source: LabelSource) -> str:
    """Human-readable identity for a LabelSource, for the image detail
    page's per-source radio buttons (see image_view_page) - mirrors
    LabelSource.__repr__'s per-source_type branching but without the
    id/debug framing, since this is user-facing UI text, not a repr."""
    if label_source.source_type == "model":
        return f"Model: {label_source.model_name}"
    if label_source.source_type == "dataset":
        return f"Dataset: {label_source.dataset.name}" if label_source.dataset else "Dataset: (unknown)"
    return f"Curator: {label_source.curator_username}"


>>>>>>> 345461a6cb7af79bfabe90381a7e2c053fd79c36
def _sha256_formatter(view, context, model, name):
    """Table cell shows only an abbreviated hash (see clean_db.abbreviate_
    hash) - the full sha256 is available via this same link's title tooltip
    and, in full, on the image detail page itself (templates/image_view.html)."""
    href = url_for("image_view_page", image_id=model.id)
    return Markup(f'<a href="{href}" title="{model.sha256}">{abbreviate_hash(model.sha256)}</a>')


class ImageView(_BaseView):
    column_list = [
        "id", "thumbnail_path", "endpoint", "training_status", "model_name",
        "sha256", "width", "height", "description", "tags",
        "label_quality_score", "inference_time_ms", "status_code",
        "client_ip", "received_at",
    ]
    column_labels = {"thumbnail_path": "Thumbnail"}
    column_formatters = {
        "thumbnail_path": _thumbnail_formatter,
        "description": _description_formatter,
        "tags": _tags_formatter,
        "sha256": _sha256_formatter,
    }
    # Inline textarea edit straight from the list view. `training_status` is
    # editable here specifically so a curator can promote a production image
    # into the trusted training corpus (or reject one) without opening the
    # full edit form - see orm.py's Image.training_status docstring; this is
    # the actual UI surface for that workflow, not incidental.
    column_editable_list = ["description", "training_status"]
    # auto_searchable_columns already covers every String/JSON column
    # (sha256, client_ip, file_path, description, capture_metadata,
    # detection_metadata, ...) - "tags.name" is added on top since it's a
    # relationship, not a column, so the auto-deriver can't see it.
    column_searchable_list = [*auto_searchable_columns(Image), "tags.name"]
    column_filters = [
        "endpoint", "training_status", "model_name", "status_code",
        "received_at", "tags", "label_quality_score",
    ]
    column_sortable_list = auto_sortable_columns(Image)
    # Left at received_at (not label_quality_score) - most rows in this table
    # are ordinary endpoint traffic with a null score (only approved/
    # training images get scored by compute_label_confidence.py), so
    # defaulting to a quality-score sort here would bury normal traffic
    # under a wall of null rows. Sorting by label_quality_score is still
    # available via column_sortable_list/the URL's ?sort= param when
    # reviewing training data specifically; training/flag_review_images.py's
    # top-5 report queries this directly rather than relying on the admin's
    # default sort.
    column_default_sort = ("received_at", True)
    # capture_metadata/detection_metadata are JSON blobs (see orm.py - one's
    # about the capture device/environment, the other's the YOLO output
    # summary) - too bulky for the list view, kept in the per-row form below.
    column_exclude_list = ["file_path", "content_type", "capture_metadata", "detection_metadata"]
    form_columns = [
        "sha256", "file_path", "width", "height", "content_type", "endpoint",
        "training_status", "model_name", "client_ip", "inference_time_ms",
        "status_code", "description", "tags", "capture_metadata",
        "detection_metadata",
    ]


class TagView(_BaseView):
    column_list = ["id", "name"]
    column_searchable_list = auto_searchable_columns(Tag)
    column_sortable_list = auto_sortable_columns(Tag)
    column_default_sort = ("name", False)
    form_columns = ["name"]


class AnnotationView(_BaseView):
    column_list = [
        "id", "image", "class_name", "confidence", "source", "label_source",
        "cross_model_agreement", "plate_text", "ocr_confidence", "region",
    ]
    column_searchable_list = auto_searchable_columns(Annotation)
    column_filters = ["class_name", "region", "source", "label_source", "cross_model_agreement"]
    column_sortable_list = auto_sortable_columns(Annotation)
    column_default_sort = ("id", False)
    form_columns = [
        "image", "class_id", "class_name", "x_center", "y_center",
        "width", "height", "confidence", "label_source", "source",
        "cross_model_agreement", "plate_text", "ocr_confidence", "region",
        "region_confidence",
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
    "image": (
        "Every image this server has ever handled - production traffic POSTed to "
        "/predict or /alpr/predict (or a frame over /alpr/ws), AND bulk-imported "
        "training-dataset images. training_status distinguishes the two for training "
        "purposes: 'unreviewed' (production default), 'approved' (safe to train on - "
        "bulk imports default here; a curator can promote a reviewed production image "
        "here too), or 'rejected'."
    ),
    "annotation": (
        "One row per detection/label tied to an image - class, box, confidence, OCR "
        "fields for license-plate detections, and which label_source produced it."
    ),
    "labelsource": (
        "Normalized identity of whatever produced a set of annotations: a specific "
        "model checkpoint (name + weights hash + major.minor version), a curated "
        "dataset, or a human curator editing a label by hand."
    ),
    "tag": 'User-curated labels (e.g. "flower", "license tag", "NYC") applied to images.',
    "dataset": "A named curated dataset (KITTI/COCO128/...) - a label_source can point at one as ground truth.",
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

    @app.route("/image/<int:image_id>/full")
    def image_full_file(image_id):
        """Serves the full-resolution original, keyed by DB id (not a raw
        filename in the URL) - looked up server-side via Image.file_path so
        the route works regardless of the on-disk extension (.jpg/.png)."""
        image = db.session.get(Image, image_id)
        if image is None:
            abort(404)
        return send_file(image.file_path)

    @app.route("/image/<int:image_id>/view")
    def image_view_page(image_id):
        """Full-size image with detection boxes/labels/OCR text drawn on a
        canvas overlay - client-side JS does the drawing (see
<<<<<<< HEAD
        templates/image_view.html) so the show/hide toggle and confidence
        slider redraw instantly with no round trip. Annotation data is
        embedded directly in the page (one query, no separate JSON
        endpoint) since this page has exactly one consumer (a human loading
        it once) - not worth a second route."""
=======
        templates/image_view.html) so the source-selection radios and
        confidence slider redraw instantly with no round trip. Annotation
        data is embedded directly in the page (one query, no separate JSON
        endpoint) since this page has exactly one consumer (a human loading
        it once) - not worth a second route.

        Every annotation is tagged with its producing label_source's
        display name (see _label_source_display_name) so the page can offer
        a radio button per distinct source - "which model/dataset/curator's
        labels am I looking at" - rather than showing every source's boxes
        overlaid on top of each other at once."""
>>>>>>> 345461a6cb7af79bfabe90381a7e2c053fd79c36
        image = db.session.get(Image, image_id)
        if image is None:
            abort(404)
        annotations = [
            {
                "class_name": a.class_name,
                # None (ground-truth/human-labeled boxes have no model
                # confidence) is normalized to 1.0 here - see the template's
                # confidence-slider docstring for why: a real model
                # confidence is always <=1.0, so treating "no confidence
                # concept applies" as the maximum means these boxes stay
                # visible at every threshold except the slider's own
                # explicit "hide everything" max value (1.01).
                "confidence": a.confidence if a.confidence is not None else 1.0,
                "x_center": a.x_center, "y_center": a.y_center,
                "width": a.width, "height": a.height,
                "plate_text": a.plate_text,
<<<<<<< HEAD
            }
            for a in image.annotations
        ]
=======
                "source": _label_source_display_name(a.label_source),
            }
            for a in image.annotations
        ]
        # Distinct sources, in first-seen order (dict preserves insertion
        # order and dedupes for free) - the radio button list.
        sources = list(dict.fromkeys(a["source"] for a in annotations))
>>>>>>> 345461a6cb7af79bfabe90381a7e2c053fd79c36
        return render_template(
            "image_view.html",
            image=image,
            image_url=url_for("image_full_file", image_id=image.id),
            annotations=annotations,
<<<<<<< HEAD
=======
            sources=sources,
>>>>>>> 345461a6cb7af79bfabe90381a7e2c053fd79c36
        )

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
        ImageView(Image, db, name="Images", category="Images & Labels"),
        AnnotationView(Annotation, db, name="Annotations", category="Images & Labels"),
        LabelSourceView(LabelSource, db, name="Label Sources", category="Images & Labels"),
        TagView(Tag, db, name="Tags", category="Images & Labels"),
        DatasetView(Dataset, db, name="Datasets", category="Images & Labels"),
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
