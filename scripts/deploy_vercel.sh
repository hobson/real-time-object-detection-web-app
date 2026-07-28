#!/usr/bin/env bash
# Deploy the Next.js in-browser ONNX detection app to Vercel (project
# yolo17/master-worktree - https://master-worktree.vercel.app).
#
# Deploys from a throwaway staging directory containing only what this app
# needs - pages/, components/, utils/, styles/, data/yolo_classes.ts, a
# small subset of models/ (see MODELS_TO_SHIP below), public/'s static
# assets, and top-level config files. NOT the repo working directory
# directly: this repo also contains inference-server/ + its .venv,
# training/ + its .venv, and data/'s datasets - multiple GB that
# `vercel --prod --archive=tgz` was found to upload wholesale
# (`.vercelignore` does not reliably apply in --archive mode in the CLI
# version this was tested with - a `vercel --prod` run from the full repo
# attempted to upload 3.8GB before failing). Building an explicit
# allowlist here sidesteps that entirely rather than trying to get an
# ignore-file denylist exactly right.
#
# public/runtime/ is deliberately NOT copied - it's a build artifact
# (next.config.js's CopyPlugin regenerates it from node_modules +
# models/). Only MODELS_TO_SHIP are included, not all seven - "Change
# Model" will 404 for whichever aren't listed there on this deployment, a
# deliberate size/speed tradeoff for this mirror, not an oversight.
#
# Not run automatically - review, then run this yourself:
#   ./scripts/deploy_vercel.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Sanity-check the build against the full repo first (catches type errors/
# build failures before spending a Vercel build minute on them) - this is
# just a local check, nothing here gets uploaded.
npm run build

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -r pages components utils styles data "$STAGE"/
mkdir -p "$STAGE/models" "$STAGE/public"
# yolo12n.onnx: RES_TO_MODEL[0], the default COCO model.
# yolo-v9-t-384-license-plate-end2end.onnx: the in-browser plate detector
# (small - 7.7MB - so including it doesn't meaningfully change deploy size).
MODELS_TO_SHIP=(yolo12n.onnx yolo-v9-t-384-license-plate-end2end.onnx)
for m in "${MODELS_TO_SHIP[@]}"; do cp "models/$m" "$STAGE/models/"; done
cp public/icon.png public/manifest.json "$STAGE/public/"
cp package.json package-lock.json next.config.js tsconfig.json \
   tailwind.config.js postcss.config.js next-env.d.ts types.d.ts \
   .eslintrc.json "$STAGE"/
rm -rf "$STAGE/data/external_datasets" "$STAGE/data/license_plates" "$STAGE/data/unlabeled"

echo "Staged $(du -sh "$STAGE" | cut -f1) across $(find "$STAGE" -type f | wc -l) files:"
du -sh "$STAGE"/* | sort -rh

cd "$STAGE"
npx vercel link --yes --project master-worktree --scope yolo17
npx vercel --prod --yes
