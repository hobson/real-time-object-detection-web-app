#!/usr/bin/env bash
# Deploy inference-server changes to taco (object-detection-inference /
# object-detection-curation systemd --user services). taco's checkout is a
# plain directory, not a git repo, so per docs/user-manual.md §4 this is a
# manual scp + restart, not a git pull.
#
# Files that differ from what's live on taco as of this writing (checked
# via `diff <(ssh taco cat ...) inference-server/<file>` for every .py file
# under inference-server/): main.py, alpr.py, curation.py, persist.py,
# orm.py, postprocess.py, and the new request_parsing.py. db.py and
# limits.py are already byte-identical and skipped.
#
# orm.py's change (added SubmittedImage.capture_metadata and
# DetectionLabel.source) needs an actual schema change on taco's already-
# existing tables - `create_all()` alone won't add columns to a table that
# already exists - so this also runs migrate_2026_07_26_capture_metadata.py
# once after copying files, before restarting the services that would
# otherwise start reading/writing those new columns.
#
# Not run automatically - review, then run this yourself:
#   ./scripts/deploy_taco.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REMOTE_DIR="~/code/hobs/real-time-object-detection-web-app/inference-server"

scp inference-server/main.py inference-server/alpr.py \
    inference-server/curation.py inference-server/persist.py \
    inference-server/orm.py inference-server/postprocess.py \
    inference-server/request_parsing.py \
    inference-server/migrate_2026_07_26_capture_metadata.py \
    inference-server/migrate_2026_07_27_thumbnail.py \
    "taco:${REMOTE_DIR}/"

ssh taco "cd ${REMOTE_DIR} && .venv/bin/python migrate_2026_07_26_capture_metadata.py && .venv/bin/python migrate_2026_07_27_thumbnail.py"

# The FastAPI process caches loaded state in memory, and gunicorn workers
# don't pick up source changes without a restart - see docs/user-manual.md
# §4 ("When deploying updated code").
ssh taco "systemctl --user restart object-detection-inference.service object-detection-curation.service"

# Confirm both came back up healthy.
ssh taco "systemctl --user is-active object-detection-inference.service object-detection-curation.service"
curl -sf https://taco.tail9f615d.ts.net:10000/infer/health && echo
curl -sf https://taco.tail9f615d.ts.net:10000/admin/ -o /dev/null -w '%{http_code}\n'
