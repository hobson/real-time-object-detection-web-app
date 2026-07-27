#!/usr/bin/env bash
# Deploy inference-server changes to taco (object-detection-inference /
# object-detection-curation systemd --user services). taco's checkout is a
# plain directory, not a git repo, so per docs/user-manual.md §4 this is a
# manual scp + restart, not a git pull.
#
# Each new orm.py column/table needs an actual schema change on taco's
# already-existing tables - `create_all()` alone only creates missing
# tables, never adds columns to one that already exists - so this runs
# every migrate_*.py script once after copying files, before restarting
# the services that would otherwise start reading/writing those columns.
# Old migrations are kept here (idempotent - every statement is
# `ADD COLUMN IF NOT EXISTS`/CREATE-if-missing) so a fresh taco checkout
# converges to the current schema in one run, not just an incremental one.
#
# Not run automatically - review, then run this yourself:
#   ./scripts/deploy_taco.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REMOTE_DIR="~/code/hobs/real-time-object-detection-web-app/inference-server"

./scripts/sync_flask_admin_toolkit.sh taco

scp inference-server/main.py inference-server/alpr.py \
    inference-server/curation.py inference-server/persist.py \
    inference-server/orm.py inference-server/postprocess.py \
    inference-server/request_parsing.py inference-server/describe.py \
    inference-server/requirements.txt \
    inference-server/migrate_2026_07_26_capture_metadata.py \
    inference-server/migrate_2026_07_27_thumbnail.py \
    inference-server/migrate_2026_07_27_description.py \
    inference-server/migrate_2026_07_27_detection_metadata_tags.py \
    "taco:${REMOTE_DIR}/"
ssh taco "mkdir -p ${REMOTE_DIR}/templates"
scp inference-server/templates/index.html "taco:${REMOTE_DIR}/templates/"

ssh taco "cd ${REMOTE_DIR} && .venv/bin/pip install -q -r requirements.txt"

ssh taco "cd ${REMOTE_DIR} && \
    .venv/bin/python migrate_2026_07_26_capture_metadata.py && \
    .venv/bin/python migrate_2026_07_27_thumbnail.py && \
    .venv/bin/python migrate_2026_07_27_description.py && \
    .venv/bin/python migrate_2026_07_27_detection_metadata_tags.py"

# The FastAPI process caches loaded state in memory, and gunicorn workers
# don't pick up source changes without a restart - see docs/user-manual.md
# §4 ("When deploying updated code").
ssh taco "systemctl --user restart object-detection-inference.service object-detection-curation.service"

# Confirm both came back up healthy.
ssh taco "systemctl --user is-active object-detection-inference.service object-detection-curation.service"
curl -sf https://taco.tail9f615d.ts.net:10000/infer/health && echo
curl -sf https://taco.tail9f615d.ts.net:10000/admin/ -o /dev/null -w '%{http_code}\n'
