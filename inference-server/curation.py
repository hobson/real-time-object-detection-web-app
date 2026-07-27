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
"""
from functools import cached_property

from flask import Flask, redirect, url_for
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.theme import Bootstrap4Theme
from flask_sqlalchemy import SQLAlchemy

from orm import (
    Base, Dataset, DatasetClass, DatasetImage, DatasetLabel,
    DetectionLabel, SubmittedImage,
)

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


class _BaseView(_CompactMixin, ModelView):
    pass


# ---------------------------------------------------------------------------
# Dataset curation views
# ---------------------------------------------------------------------------

class DatasetView(_BaseView):
    column_list = ["id", "name", "description", "created_at"]
    column_searchable_list = ["name", "description"]
    column_sortable_list = ["id", "name", "created_at"]
    column_default_sort = ("created_at", True)
    form_columns = ["name", "description"]


class DatasetClassView(_BaseView):
    column_list = ["id", "dataset", "class_index", "name"]
    column_searchable_list = ["name"]
    column_filters = ["dataset"]
    column_sortable_list = ["id", "class_index", "name"]
    column_default_sort = ("class_index", False)
    form_columns = ["dataset", "class_index", "name"]


class DatasetImageView(_BaseView):
    column_list = ["id", "dataset", "file_path", "split", "width", "height", "created_at"]
    column_searchable_list = ["file_path"]
    column_filters = ["dataset", "split"]
    column_sortable_list = ["id", "file_path", "split", "created_at"]
    column_default_sort = ("created_at", True)
    column_editable_list = ["split"]
    form_columns = ["dataset", "file_path", "width", "height", "split"]


class DatasetLabelView(_BaseView):
    column_list = [
        "id", "image", "class_index", "x_center", "y_center", "width", "height",
    ]
    column_filters = ["class_index"]
    column_sortable_list = ["id", "class_index"]
    column_default_sort = ("id", False)
    form_columns = ["image", "class_index", "x_center", "y_center", "width", "height"]


# ---------------------------------------------------------------------------
# Endpoint logging views
# ---------------------------------------------------------------------------

class SubmittedImageView(_BaseView):
    column_list = [
        "id", "endpoint", "model_name", "sha256", "width", "height",
        "inference_time_ms", "status_code", "client_ip", "received_at",
    ]
    column_searchable_list = ["sha256", "client_ip"]
    column_filters = ["endpoint", "model_name", "status_code", "received_at"]
    column_sortable_list = ["id", "endpoint", "inference_time_ms", "received_at"]
    column_default_sort = ("received_at", True)
    # capture_metadata is a JSON blob (GPS/orientation/acceleration/camera-
    # facing) - too bulky for the list view, kept in the per-row form below.
    column_exclude_list = ["file_path", "content_type", "capture_metadata"]
    form_columns = [
        "sha256", "file_path", "width", "height", "content_type", "endpoint",
        "model_name", "client_ip", "inference_time_ms", "status_code",
        "capture_metadata",
    ]


class DetectionLabelView(_BaseView):
    column_list = [
        "id", "submitted_image", "class_name", "confidence", "source",
        "plate_text", "ocr_confidence", "region", "model_name",
    ]
    column_searchable_list = ["class_name", "plate_text", "model_name"]
    column_filters = ["class_name", "model_name", "region", "source"]
    column_sortable_list = ["id", "confidence", "ocr_confidence"]
    column_default_sort = ("id", False)
    form_columns = [
        "submitted_image", "class_id", "class_name", "x_center", "y_center",
        "width", "height", "confidence", "model_name", "source", "plate_text",
        "ocr_confidence", "region", "region_confidence",
    ]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

class _HomeView(_CompactMixin, AdminIndexView):
    """Redirect /admin/ to the submitted-images table; inject compact CSS."""

    @expose("/")
    def index(self):
        return redirect(url_for("submittedimage.index_view"))


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

    # Tables are created once via `python orm.py` (see docs/user-manual.md
    # §3), not here - this avoids every gunicorn worker re-running DDL
    # schema checks against Postgres on every startup/restart.
    db = SQLAlchemy(app, model_class=Base)

    admin = Admin(
        app,
        name="Inference Server Curation",
        theme=Bootstrap4Theme(swatch="flatly"),
        index_view=_HomeView(url="/"),
    )

    admin.add_view(SubmittedImageView(SubmittedImage, db, name="Submitted Images", category="Endpoint Traffic"))
    admin.add_view(DetectionLabelView(DetectionLabel, db, name="Detection Labels", category="Endpoint Traffic"))
    admin.add_view(DatasetView(Dataset, db, name="Datasets", category="Dataset Curation"))
    admin.add_view(DatasetClassView(DatasetClass, db, name="Dataset Classes", category="Dataset Curation"))
    admin.add_view(DatasetImageView(DatasetImage, db, name="Dataset Images", category="Dataset Curation"))
    admin.add_view(DatasetLabelView(DatasetLabel, db, name="Dataset Labels", category="Dataset Curation"))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
