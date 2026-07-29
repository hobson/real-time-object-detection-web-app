# Inference API + data curation admin — user manual

`inference-server/` runs two separate apps on the host `taco`, both public
at `https://taco.tail9f615d.ts.net:10000/` alongside the existing in-browser
detection app (see [`realtime-object-detection.md`](./realtime-object-detection.md)
for that one — it lives at the root `/` path and is unaffected by anything
in this doc):

| Path | App | Purpose |
|---|---|---|
| `/infer/*` | FastAPI (`main.py` + `alpr.py`) | Server-side YOLO detection + license-plate detection/OCR, callable over HTTP/WebSocket |
| `/admin/*` | Flask-Admin (`curation.py`) | Browse/edit everything the inference API has logged, plus curate standalone YOLO training datasets |

Both are backed by a Postgres database (`orm.py`'s tables) that records
**every image submitted to `/infer/predict` and `/infer/alpr/*`, and every
detection returned for it** — so nothing analyzed through the API is lost;
it all becomes reviewable/curatable data.

See [`system-architecture.md`](./system-architecture.md) for a network-level
diagram of how these pieces (and the frontend, Tailscale Funnel routing, and
training pipeline) fit together.

## 1. Calling the inference API (`/infer`)

All paths below are relative to `https://taco.tail9f615d.ts.net:10000/infer`.
For local development, run the server yourself (§3) and drop the `/infer`
prefix (e.g. `http://localhost:8092/predict`) since the prefix is only added
by taco's reverse-proxy path routing, not by the app itself.

### Interactive docs

FastAPI generates interactive API docs automatically from the route
definitions — no separate doc-writing to keep in sync:

- **`GET /infer/docs`** — Swagger UI, with a "Try it out" button per
  endpoint (works for `/predict`/`/alpr/predict` too — pick a file, hit
  Execute).
- **`GET /infer/redoc`** — ReDoc, a read-only single-page alternative.
- **`GET /infer/openapi.json`** — the raw OpenAPI schema, if you're
  generating a client.

```
https://taco.tail9f615d.ts.net:10000/infer/docs
```

### `GET /health`

```bash
curl https://taco.tail9f615d.ts.net:10000/infer/health
# {"status":"ok"}
```

### `GET /models`

Lists the general-purpose (COCO, 80-class) YOLO model variants available to
`/predict`:

```bash
curl https://taco.tail9f615d.ts.net:10000/infer/models
# {"models":["yolo12n.onnx","yolo11n.onnx","yolov10n.onnx",
#   "yolov7-tiny_256x256.onnx","yolov7-tiny_320x320.onnx",
#   "yolov7-tiny_640x640.onnx"],"default":"yolo12n.onnx"}
```

### `POST /predict?model=<name>` — general object detection

Body is the raw image bytes (JPEG or PNG); `model` query param is optional
(defaults to `yolo12n.onnx`, must be one of the names from `/models`).

```bash
curl -X POST "https://taco.tail9f615d.ts.net:10000/infer/predict?model=yolo12n.onnx" \
  --data-binary @photo.jpg -H "Content-Type: image/jpeg"
```

```json
{
  "model": "yolo12n.onnx",
  "inferenceTimeMs": 28.7,
  "detections": [
    {"class": "car", "confidence": 0.8253, "box": [0.084, 0.133, 0.874, 0.816]}
  ]
}
```

`box` is `[x0, y0, x1, y1]`, normalized 0–1 against the image's own
width/height (not the model's internal input resolution) — multiply by your
image dimensions to get pixel coordinates.

### Attaching capture metadata (multipart)

Both `/predict` and `/alpr/predict` also accept `multipart/form-data`
instead of a raw body, for clients that have more than just the image to
send — an `image` file part plus an optional `metadata` JSON string part:

```bash
curl -X POST "https://taco.tail9f615d.ts.net:10000/infer/predict" \
  -F "image=@photo.jpg;type=image/jpeg" \
  -F 'metadata={"gps":{"lat":37.7749,"lon":-122.4194,"accuracy":8.0},"camera_facing":"environment"}'
```

`metadata` is a free-form JSON object — nothing in it is validated beyond
"is this a JSON object" — but the fields a client is expected to send are:

| Field | Shape | Source |
|---|---|---|
| `gps` | `{lat, lon, altitude?, accuracy?, heading?, speed?}` | `navigator.geolocation` |
| `orientation` | `{alpha, beta, gamma}` | `DeviceOrientationEvent` (compass heading + tilt) |
| `acceleration` | `{x, y, z}` | `DeviceMotionEvent.acceleration` |
| `rotation_rate` | `{alpha, beta, gamma}` | `DeviceMotionEvent.rotationRate` (gyroscope) |
| `camera_facing` | `"user"` \| `"environment"` | which physical camera captured the frame |
| `client_detections` | same shape as this endpoint's own `detections` response field | the client's own YOLO results, if it ran detection itself before/instead of relying on this call |

Everything is optional and stored as-is (`SubmittedImage.capture_metadata`,
a JSON column — see §2's Submitted Images detail view). `client_detections`
specifically is pulled out and stored as `DetectionLabel` rows like any
other detection, but with `source="client"` instead of the default
`"server"`, so `/admin`'s Detection Labels table can tell the two apart
(filter by `source`). None of this changes the response shape — a
multipart request gets back exactly the same JSON either endpoint would
return for a plain raw-body request.

### `POST /alpr/predict` — license plate detection + OCR

Body is a single JPEG frame (raw bytes). No `model` param — always uses the
plate detector + OCR model pair configured via `ALPR_DETECTOR_MODEL`/
`ALPR_OCR_MODEL` env vars (defaults: `yolo-v9-t-384-license-plate-end2end` /
`cct-xs-v2-global-model`).

```bash
curl -X POST "https://taco.tail9f615d.ts.net:10000/infer/alpr/predict" \
  --data-binary @plate.jpg -H "Content-Type: image/jpeg"
```

```json
{
  "inferenceTimeMs": 24.9,
  "detections": [
    {
      "box": [0.593, 0.565, 0.747, 0.634],
      "detectionConfidence": 0.843,
      "plate": "5AU5341",
      "ocrConfidence": 0.9998,
      "region": "Czech Republic",
      "regionConfidence": 1.0
    }
  ]
}
```

`plate`/`ocrConfidence`/`region`/`regionConfidence` are `null` when a plate
box is detected but OCR couldn't read it confidently. Region classification
is best-effort and not always accurate for plate styles the model wasn't
trained on.

### `WS /alpr/ws` — streaming license plate detection

For continuous video rather than one-off images: open a WebSocket, send one
binary JPEG frame per message, get back one JSON detections message per
frame, in order. Designed for ~1 fps client pacing (see `alpr.py`'s module
docstring for why WebSocket over HTTP-polling or WebRTC at that rate).

```python
import asyncio, json
import websockets

async def main():
    async with websockets.connect(
        "wss://taco.tail9f615d.ts.net:10000/infer/alpr/ws"
    ) as ws:
        for jpeg_bytes in frames:  # your own frame source
            await ws.send(jpeg_bytes)
            result = json.loads(await ws.recv())
            print(result["detections"])

asyncio.run(main())
```

A malformed frame gets back `{"error": "..."}` on that message only — the
connection stays open and keeps processing subsequent frames.

### What gets recorded

Every successful call to `/predict`, `/alpr/predict`, or `/alpr/ws`
persists the submitted image (deduplicated by SHA256 — sending the same
bytes twice doesn't store two copies) plus one row per detection returned,
best-effort (a database hiccup logs a warning server-side but never fails
your request). See §2 to browse what's been recorded.

## 2. Browsing/curating data (`/admin`)

Open **`https://taco.tail9f615d.ts.net:10000/admin/`** in a browser. No
login — it's not intended to be shared beyond people who already have the
funnel URL.

### Endpoint Traffic

- **Submitted Images** — every image sent to the API (§1), with endpoint,
  model, dimensions, timing, and client IP. Click a row to see its detail
  form; the raw image bytes live on taco's disk at the shown `file_path`,
  not in the browser.
- **Detection Labels** — every detection returned for those images. Filter
  by `class_name`/`model_name`/`region`; plate-specific columns
  (`plate_text`, `ocr_confidence`, `region`) are blank for non-ALPR
  detections.

<<<<<<< HEAD
=======
### Correcting machine-labeled detections

The model's output isn't locked in — every field it wrote is a normal
database row you can fix by hand. There's no "confirm"/"reject" workflow;
editing a label in place *is* the correction.

1. Find the image in **Submitted Images** (search by `sha256` or filter by
   `endpoint`/`received_at`) and open its `file_path` on taco's disk to see
   what was actually submitted — the admin list doesn't render the image
   inline.
2. Go to **Detection Labels** and filter by `submitted_image` to see just
   that image's detections (a `SubmittedImage` row doesn't embed its
   detections directly — they're a separate, filterable table because one
   image can have many).
3. Fix a wrong reading directly:
   - Misread plate text or wrong class → click-to-edit `plate_text` /
     `class_name` in the list view, or open the row's full edit form for
     everything else (`confidence`, box coordinates, `region`,
     `region_confidence`).
   - A phantom detection (box shouldn't exist at all) or a missed plate
     (no box was drawn) → delete the bogus row, or add a new
     `DetectionLabel` row by hand with `submitted_image` set to the image
     it belongs to.
4. Corrections here only fix the historical record for review/export — they
   don't retrain or otherwise feed back into the live `ALPR_DETECTOR_MODEL`/
   `ALPR_OCR_MODEL` models serving new requests. To build a training set from
   corrected examples, copy them into **Dataset Curation** (below) instead.

>>>>>>> 00189ca3dedb7f88ca31d958deec2e78b2e52c7f
### Dataset Curation

A separate, generic YOLO dataset schema — not automatically populated from
endpoint traffic, and not tied to this repo's own `data/license_plates/`
on-disk dataset. Use it to hand-curate a training set independently:

1. Create a **Dataset** (just a name + description).
2. Add its **Dataset Classes** (class index + name — your own numbering,
   doesn't have to match COCO's).
3. Add **Dataset Images** (a `file_path` you manage yourself — this table
   doesn't upload/store image bytes) with a `split` (train/val/test).
4. Add **Dataset Labels** (normalized YOLO boxes) against each image.

### UI tips

- **Click-to-edit**: click most cells directly in the list view rather than
  opening the full edit form — it saves that one field immediately.
- **Ctrl+Enter / Cmd+Enter** saves the current inline edit or form without
  reaching for the mouse.
- Column headers are clickable to sort; the search box and filter dropdowns
  above each table narrow the list without editing anything.

## 3. Running it yourself (local development)

```bash
cd inference-server
pip install -r requirements.txt -r requirements-curation.txt
cp .env.example .env   # then edit DATABASE_URL to point at your own Postgres
python orm.py           # creates the tables (idempotent, safe to re-run)

uvicorn main:app --port 8092 &     # the FastAPI inference API
gunicorn -w 2 -b 127.0.0.1:8093 curation:app &   # the Flask-Admin curation app
```

- FastAPI app: `http://localhost:8092/health`, `/predict`, `/alpr/predict`, `/alpr/ws`.
- Curation app: `http://localhost:8093/` (mounted at its own root locally —
  the `/admin` prefix only appears in production because taco's reverse
  proxy adds it, see §4).
- `orm.py` defaults to a Postgres URL if `DATABASE_URL` isn't set, but
  `create_engine` accepts any SQLAlchemy URL — a `sqlite:///path/to.db`
  works fine for quick local testing without standing up Postgres.
- `requirements-curation.txt` is separate from `requirements.txt`
  specifically so a plain FastAPI-only deployment doesn't need
  Flask/Flask-Admin/gunicorn at all.

## 4. Operating the taco deployment

Two `systemd --user` services on taco:

```bash
ssh taco systemctl --user status object-detection-inference.service   # FastAPI, port 8092
ssh taco systemctl --user status object-detection-curation.service    # Flask-Admin, port 8093
ssh taco journalctl --user -u object-detection-inference.service -f   # tail logs
ssh taco systemctl --user restart object-detection-inference.service  # after deploying new code
```

Both are backed by a local Postgres database `inference_server` on taco
(peer-auth over the Unix socket, no password — see `inference-server/.env`
on taco for the exact `DATABASE_URL`).

Path routing on port 10000 is `tailscale funnel` with `--set-path`, which
**strips the matched prefix before proxying** — the FastAPI/Flask apps
underneath have no idea they're mounted at `/infer`/`/admin` externally,
they just see requests at their own root. This is transparent for ordinary
routes (`/health`, `/predict`, ...), but FastAPI's auto-generated docs page
embeds an absolute `/openapi.json` URL in its HTML unless told otherwise —
without the fix below, `/infer/docs` loads but shows "Failed to load API
definition" because the browser fetches `/openapi.json` (hitting the root
WASM app) instead of `/infer/openapi.json`. Fixed by passing uvicorn
`--root-path /infer` (see the `ExecStart` line in
`object-detection-inference.service`), which only affects URLs FastAPI
generates for the client (docs links, the `servers` field in the OpenAPI
schema) — it does not change how incoming request paths are matched, so
this flag must stay in sync with whatever `--set-path` value is actually
funneled to this service.

The Flask-Admin app has the same problem, since it's mounted at its own
root: `url_for()` (used for the home-page redirect to `/submittedimage/`,
and for every static asset link) generates paths without the `/admin`
prefix, which 404 when the browser resolves them against the real
(unstripped) domain. Fixed the same way, via `ADMIN_ROOT_PATH=/admin` (see
the `Environment=` line in `object-detection-curation.service`) — it sets
WSGI's `SCRIPT_NAME` so Flask's own URL generation stays prefix-aware. Like
`--root-path` above, this must stay in sync with whatever path
`tailscale funnel` actually maps to this service:

```bash
ssh taco tailscale funnel status
```
```
https://taco.tail9f615d.ts.net:10000 (Funnel on)
|-- /      proxy http://127.0.0.1:3101   # the WASM app (unrelated to this doc)
|-- /infer proxy http://127.0.0.1:8092   # main.py + alpr.py
|-- /admin proxy http://127.0.0.1:8093   # curation.py
```

**To add or change a path mapping**, use `tailscale funnel`, not
`tailscale serve` — `serve` only shares within the tailnet, and running it
against a port that's already a public `funnel` silently **downgrades that
whole port to tailnet-only**, taking down public access to everything
already mapped there (including the unrelated `/` WASM app). Always
`tailscale funnel status` before and after any change to confirm it still
says `Funnel on`, not `(tailnet only)`.

```bash
ssh taco tailscale funnel --bg --https=10000 --set-path=/infer http://127.0.0.1:8092
```

**When deploying updated code**: this repo's clone on taco is separate from
any other machine's — code changes need to be copied over (`scp`/tar, not
`rsync` — blocked under this project's automation) and the relevant
`systemctl --user restart` run afterward. Two things are easy to forget
because they're outside `inference-server/`'s own directory:
- The `.onnx` model files live in `../models/`, not `inference-server/` —
  re-export or copy those too if they changed, and restart afterward (the
  FastAPI app caches loaded ONNX sessions in memory, so a stale model stays
  loaded until the process restarts even after the file on disk changes).
- Never copy your own local `.env` to taco — it points at *your* Postgres,
  not taco's. taco has its own `.env` with its own `DATABASE_URL`.
