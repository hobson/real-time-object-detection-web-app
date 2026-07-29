#!/usr/bin/env bash
# Shared aliveness-check helper for this repo's deploy scripts
# (deploy_vercel.sh, deploy_taco_frontend.sh) - a plain `curl -sf` GET per
# URL, printing status code. Source this file, then call check_alive with
# one or more URLs.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/check_alive.sh"
#   check_alive "https://example.com/" "https://example.com/health"
check_alive() {
  for url in "$@"; do
    curl -sf -o /dev/null -w "GET ${url} -> %{http_code}\n" "$url"
  done
}
