#!/usr/bin/env bash
# Deploy the Next.js in-browser ONNX detection app to Vercel (project
# yolo17/master-worktree - https://master-worktree.vercel.app).
#
# .vercelignore in this repo trims the upload to just what this app needs:
# it deliberately excludes inference-server/, training/, data/'s dataset
# subdirs, and all models/*.onnx except yolo12n.onnx (the default model -
# see .vercelignore's comments for the size tradeoff that implies).
#
# Not run automatically - review the plan below, then run this yourself:
#   ./deploy_vercel.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Sanity-check the build locally first (catches type errors / build
# failures before spending a Vercel build minute on them).
npm run build

# Link this directory to the existing yolo17/master-worktree project if it
# isn't already (safe to re-run - no-ops once .vercel/ exists).
npx vercel link --yes --project master-worktree --scope yolo17

# --archive=tgz avoids the CLI's ~15,000-file upload cap; --prod publishes
# straight to the production URL (master-worktree.vercel.app) rather than
# a preview deployment.
npx vercel --prod --yes --archive=tgz
