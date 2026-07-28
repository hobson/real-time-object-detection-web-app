#!/usr/bin/env bash
# Deploy the Next.js frontend to taco's self-hosted mirror
# (object-detection-web.service, npm start on :3101, proxied at path "/" -
# see docs/system-architecture.md). This is a THIRD deploy target distinct
# from both Vercel (scripts/deploy_vercel.sh) and taco's inference-server
# (scripts/deploy_taco.sh) - easy to forget since it's the same taco host,
# but it's a separate systemd service running from the same repo checkout
# there, and needs its own rebuild + restart whenever frontend code
# (components/, pages/, utils/, data/yolo_classes.ts, models/) changes.
#
# taco's checkout is a plain directory, not a git repo (see
# docs/user-manual.md §4), so this is tar-over-ssh, not a git pull, and
# explicitly not rsync (blocked under this project's automation).
#
# Not run automatically - review, then run this yourself:
#   ./scripts/deploy_taco_frontend.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Sanity-check the build locally first.
npm run build

REMOTE_DIR="~/code/hobs/real-time-object-detection-web-app"

# Only source the frontend actually needs - NOT data/ wholesale (its
# dataset subdirs are hundreds of MB and irrelevant to the frontend; only
# data/yolo_classes.ts is ever imported), NOT node_modules/.next (rebuilt
# on taco), NOT inference-server/training (separate deploy targets).
tar czf - \
    pages components utils styles models \
    data/yolo_classes.ts \
    public/icon.png public/manifest.json \
    package.json package-lock.json next.config.js tsconfig.json \
    tailwind.config.js postcss.config.js next-env.d.ts types.d.ts \
    .eslintrc.json \
  | ssh taco "cd ${REMOTE_DIR} && tar xzf -"

ssh taco "cd ${REMOTE_DIR} && PATH=\$HOME/.nvm/versions/node/v24.18.0/bin:\$PATH npm install && PATH=\$HOME/.nvm/versions/node/v24.18.0/bin:\$PATH npm run build"

ssh taco "systemctl --user restart object-detection-web.service"
ssh taco "systemctl --user is-active object-detection-web.service"
curl -sf -o /dev/null -w "GET / -> %{http_code}\n" https://taco.tail9f615d.ts.net:10000/
curl -sf -o /dev/null -w "GET /runtime/yolo-v9-t-384-license-plate-end2end.onnx -> %{http_code}\n" \
  https://taco.tail9f615d.ts.net:10000/runtime/yolo-v9-t-384-license-plate-end2end.onnx
