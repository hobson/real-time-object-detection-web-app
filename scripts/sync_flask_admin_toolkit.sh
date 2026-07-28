#!/usr/bin/env bash
# Sync ~/code/hobs/flask-admin-toolkit (curation.py's search/sort
# dependency, see inference-server/requirements.txt's
# `-e ../../flask-admin-toolkit`) to a remote host. It's a sibling package,
# not published anywhere, so any deploy that installs inference-server's
# requirements needs this synced first - factored out of deploy_taco.sh so
# a future deploy target needing the same toolkit doesn't have to
# reimplement this.
#
# Usage: ./scripts/sync_flask_admin_toolkit.sh <host> [remote_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOST="${1:?Usage: $0 <host> [remote_dir]}"
REMOTE_DIR="${2:-~/code/hobs/flask-admin-toolkit}"

ssh "${HOST}" "mkdir -p ${REMOTE_DIR}"
scp -r ../flask-admin-toolkit/pyproject.toml ../flask-admin-toolkit/README.md ../flask-admin-toolkit/src \
    "${HOST}:${REMOTE_DIR}/"
