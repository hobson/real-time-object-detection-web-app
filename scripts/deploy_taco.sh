#!/usr/bin/env bash
# Deploy inference-server changes to taco (object-detection-inference /
# object-detection-curation systemd --user services). taco's checkout is a
# plain directory, not a git repo, so per docs/user-manual.md §4 this is a
# manual scp + restart, not a git pull.
#
# As of this writing, only two files differ from what's already live on
# taco (checked via `diff <(ssh taco cat ...) inference-server/<file>`):
#   - inference-server/alpr.py     (deduped a comment)
#   - inference-server/curation.py (cached_property instead of property)
# Everything else under inference-server/ (db.py, limits.py, main.py,
# persist.py, orm.py, ...) is already byte-identical on taco.
#
# Not run automatically - review, then run this yourself:
#   ./deploy_taco.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REMOTE_DIR="~/code/hobs/real-time-object-detection-web-app/inference-server"

scp inference-server/alpr.py inference-server/curation.py \
    "taco:${REMOTE_DIR}/"

# The FastAPI process caches loaded state in memory, and gunicorn workers
# don't pick up source changes without a restart - see docs/user-manual.md
# §4 ("When deploying updated code").
ssh taco "systemctl --user restart object-detection-inference.service object-detection-curation.service"

# Confirm both came back up healthy.
ssh taco "systemctl --user is-active object-detection-inference.service object-detection-curation.service"
curl -sf https://taco.tail9f615d.ts.net:10000/infer/health && echo
curl -sf https://taco.tail9f615d.ts.net:10000/admin/ -o /dev/null -w '%{http_code}\n'
