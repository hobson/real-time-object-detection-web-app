"""Download YOLO-format dataset repos from the Hugging Face Hub.

Fetches plain dataset repos (images + YOLO-format label .txt files), not
`datasets`-library datasets, so this uses `snapshot_download` rather than
`load_dataset`. Before downloading, it scans the local Hub cache
(`~/.cache/huggingface/hub` by default, or `$HF_HOME`/`$HUGGINGFACE_HUB_CACHE`
if set) and reports what it finds - `snapshot_download` itself is cache-aware
and re-fetches only files that changed, but this makes that reuse visible
instead of silent.

Usage:
    python fetch_hf_dataset.py <repo_id> [<repo_id> ...] [--local-dir-base DIR]

Example:
    python fetch_hf_dataset.py FACADEEEE/yolo-license-plates-dataset \
        Egzavyer/kitti-yolo LibreYOLO/coco128
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HFCacheInfo, scan_cache_dir, snapshot_download

DEFAULT_LOCAL_DIR_BASE = Path(__file__).resolve().parent.parent / "data" / "external_datasets"


def fetch(repo_id: str, local_dir_base: Path, repo_type: str, max_workers: int,
          cache: HFCacheInfo) -> Path:
    cached = next((r for r in cache.repos if r.repo_id == repo_id and r.repo_type == repo_type), None)
    if cached:
        print(f"[{repo_id}] found in Hub cache at {cached.repo_path} "
              f"({cached.size_on_disk / 1_000_000:.1f} MB) - will reuse unchanged files")
    else:
        print(f"[{repo_id}] not in Hub cache - will download fresh")

    local_dir = local_dir_base / repo_id.split("/")[-1]
    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir,
                              max_workers=max_workers)
    print(f"[{repo_id}] available at {path}")
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_ids", nargs="+", help="e.g. FACADEEEE/yolo-license-plates-dataset")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir-base", type=Path, default=DEFAULT_LOCAL_DIR_BASE,
                         help=f"default: {DEFAULT_LOCAL_DIR_BASE}")
    parser.add_argument("--max-workers", type=int, default=16,
                         help="per-repo download concurrency, higher helps repos with many "
                              "small files where per-file HTTP overhead dominates (default: 16)")
    args = parser.parse_args()

    cache = scan_cache_dir()
    with ThreadPoolExecutor(max_workers=len(args.repo_ids)) as pool:
        pool.map(lambda repo_id: fetch(repo_id, args.local_dir_base, args.repo_type,
                                        args.max_workers, cache),
                 args.repo_ids)


if __name__ == "__main__":
    main()
