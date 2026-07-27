"""Generate accessibility-style image descriptions via a colocated
multimodal LLM (llama-swap running llama.cpp's server, e.g. on taco) and
persist them onto SubmittedImage.description.

Why a background queue instead of doing this inline with /predict: a
vision-LLM call is 1-2 orders of magnitude slower than the YOLO inference
this server otherwise does (a model swap alone, if the vision model isn't
already loaded on llama-swap's single GPU slot - see CLAUDE.md's
llama-swap notes - can take longer than the healthCheckTimeout it's
configured with), so doing it inline would make every capture as slow as
the LLM. main.py/alpr.py instead enqueue a description job after
persisting and return the detection response immediately; a background
asyncio task drains the queue and writes the result back once it's ready.

Why the queue drops instead of buffering: `/predict` and `/alpr/predict`
can be called many times a second by a live-detection loop
(ObjectDetectionCamera's requestAnimationFrame loop, or AlprServer's ~1fps
stream). Enqueuing every single frame would both build an ever-growing
backlog (production far outpaces one-at-a-time LLM consumption) and thrash
llama-swap's single-model GPU slot against whatever else is using it (see
llama-swap.yaml's "one model loaded at a time" design). A small
bounded queue with drop-on-full semantics keeps this to "best-effort
caption for an occasional/representative capture," which is all a
description field needs - not "caption every frame."

CLI usage (from inference-server/, same venv as main.py) - backfill
descriptions for every already-stored image that doesn't have one yet
(e.g. images captured before this feature existed, or where a previous
background job attempt failed):

    python describe.py
    python describe.py --limit 50
"""
import argparse
import asyncio
import base64
import logging
import os
from pathlib import Path

import httpx

from db import SessionLocal
from orm import SubmittedImage

logger = logging.getLogger("describe")

LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
LLAMA_VISION_MODEL = os.environ.get("LLAMA_VISION_MODEL", "gemma-3-4b-it-vision")
LLAMA_TIMEOUT_S = float(os.environ.get("LLAMA_TIMEOUT_S", "180"))

# See module docstring - bounds the background queue so a burst of frames
# from a live-detection loop can't build an unbounded backlog.
QUEUE_MAXSIZE = int(os.environ.get("DESCRIBE_QUEUE_MAXSIZE", "2"))

PROMPT = (
    "Describe this image the way accessibility alt text would: one or two "
    "concise, factual sentences naming the scene and the main subjects/"
    "objects, suitable for a screen reader. On a new line after that, add "
    "'Keywords: ' followed by 5-15 comma-separated keywords naming the "
    "objects, scene type, and setting visible. Only describe what is "
    "actually visible - do not speculate."
)


def generate_description(image_bytes: bytes, *, content_type: str = "image/jpeg") -> str:
    """Call the colocated vision model and return its generated
    description. Raises on any failure (HTTP error, timeout, unexpected
    response shape) - callers that want best-effort behavior should catch,
    same contract as persist.py's own best-effort functions."""
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"} if LLAMA_API_KEY else {}

    response = httpx.post(
        f"{LLAMA_BASE_URL}/v1/chat/completions",
        headers=headers,
        timeout=LLAMA_TIMEOUT_S,
        json={
            "model": LLAMA_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _update_description(submitted_image_id: int, description: str) -> None:
    session = SessionLocal()
    try:
        row = session.get(SubmittedImage, submitted_image_id)
        if row is not None:
            row.description = description
            session.commit()
    finally:
        session.close()


def describe_and_store(
    submitted_image_id: int, image_bytes: bytes, *, content_type: str = "image/jpeg"
) -> bool:
    """Best-effort: generate + persist a description for one already-stored
    SubmittedImage row. Never raises - both the background worker (in a
    thread) and the CLI backfill below call this, and a slow/failed LLM
    call must not crash either. Returns whether it succeeded, so callers
    that want to report per-row status (the CLI) don't need their own
    try/except around the same two calls."""
    try:
        description = generate_description(image_bytes, content_type=content_type)
        _update_description(submitted_image_id, description)
        return True
    except Exception:
        logger.exception(
            "Failed to generate description for submitted_image %s", submitted_image_id
        )
        return False


# ---------------------------------------------------------------------------
# Background queue (used by main.py/alpr.py) - see module docstring
# ---------------------------------------------------------------------------

# Constructing an asyncio.Queue no longer requires a running loop as of
# Python 3.10, so this can be a plain module-level singleton instead of a
# lazily-initialized one.
_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)


def enqueue_description(
    submitted_image_id: int | None, image_bytes: bytes, content_type: str | None
) -> None:
    """Non-blocking: silently drops the job if the queue is full (see
    module docstring) or if persistence itself failed upstream (no row to
    attach a description to)."""
    if submitted_image_id is None:
        return
    try:
        _queue.put_nowait((submitted_image_id, image_bytes, content_type or "image/jpeg"))
    except asyncio.QueueFull:
        logger.debug(
            "Description queue full, dropping submitted_image %s", submitted_image_id
        )


async def _worker() -> None:
    while True:
        submitted_image_id, image_bytes, content_type = await _queue.get()
        try:
            await asyncio.to_thread(
                describe_and_store, submitted_image_id, image_bytes, content_type=content_type
            )
        finally:
            _queue.task_done()


def start_worker() -> asyncio.Task:
    """Call once from an app's startup hook (see main.py)."""
    return asyncio.create_task(_worker())


# ---------------------------------------------------------------------------
# CLI: backfill descriptions for existing rows
# ---------------------------------------------------------------------------

def backfill_missing(limit: int | None = None) -> None:
    session = SessionLocal()
    try:
        query = (
            session.query(SubmittedImage)
            .filter(SubmittedImage.description.is_(None))
            .order_by(SubmittedImage.id)
        )
        if limit:
            query = query.limit(limit)
        rows = query.all()
    finally:
        session.close()

    print(f"Found {len(rows)} submitted image(s) without a description")
    for i, row in enumerate(rows, 1):
        path = Path(row.file_path)
        if not path.exists():
            print(f"[{i}/{len(rows)}] submitted_image {row.id}: skipped, file missing at {path}")
            continue
        print(f"[{i}/{len(rows)}] submitted_image {row.id} ({path.name}) ...", end=" ", flush=True)
        ok = describe_and_store(
            row.id, path.read_bytes(), content_type=row.content_type or "image/jpeg"
        )
        print("ok" if ok else "failed (see log)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process at most N images")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    backfill_missing(limit=args.limit)


if __name__ == "__main__":
    main()
