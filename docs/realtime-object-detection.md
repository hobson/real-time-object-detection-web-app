# Using real-time object detection on your phone

This app runs YOLO object detection on your phone's camera feed, right in
the browser — no app install required. It can detect 80 everyday object
classes (person, car, bicycle, dog, etc. — see
[`data/yolo_classes.ts`](../data/yolo_classes.ts) for the full list) and
optionally push a notification with a photo when it sees a person or car.

## 1. Open the app

Visit one of these URLs on your phone:

- **`https://taco.tail9f615d.ts.net:10000/`** — recommended; self-hosted, no usage limits.
- **`https://master-worktree.vercel.app/`** — Vercel-hosted mirror of the same client-side app.

Both require **HTTPS** — the browser will refuse camera access otherwise.
If you're testing a local dev build instead, see the note at the bottom.

## 2. Grant camera permission

The browser will prompt for camera access on first load. Allow it — the
app can't do anything without it, and nothing is uploaded anywhere unless
you're using notifications (see §5) or Server-side mode (see §4).

## 3. Wait for "Loading model..." to finish

Before you can detect anything, the app needs to either download a model
to your phone (In-browser mode) or confirm it can reach the inference
server (Server-side mode). While that's happening, the **Capture Photo**
and **Live Detection** buttons are disabled and show a spinner:

> Loading model (attempt 1 of 5)... (118s)

**This can take a while on a slow connection** — the in-browser model
is a one-time ~10-25MB download, and if you're on the self-hosted
(`taco.tail9f615d.ts.net`) URL, that traffic is relayed through
Tailscale's infrastructure rather than a direct connection, which is
measurably slow for large files. Don't worry if the countdown runs out —
it automatically retries with backoff, up to 5 attempts, and tells you
which attempt it's on:

> Attempt 2 of 5 failed, retrying in 6s...

If all 5 attempts fail, you'll see a red error message describing what
went wrong and a **Retry** button. Common causes: you lost signal/Wi-Fi
mid-download, or (rarer) the server itself is down — retrying a few
times, or switching to a better connection, usually resolves it.

Once ready, the buttons switch to their normal labels and are enabled.

## 4. Pick a mode: In-browser vs Server-side

Two buttons at the top of the page switch between them:

- **In-browser (WASM)** — the default. Inference runs entirely on your
  phone; nothing about what your camera sees ever leaves the device
  (except captured/notified frames, see §5). Slower to start (has to
  download the model first, once per session) but works even if the
  server is temporarily unreachable, once loaded.
- **Server-side** — your phone just captures and uploads a JPEG frame per
  detection; a server does the actual inference and sends back the
  results. Starts almost instantly (just checks the server is reachable,
  no big download) and tends to run smoother frame-to-frame, at the cost
  of every analyzed frame being sent to the server.

Switching modes re-triggers the loading step in §3 for whichever mode you
switched to.

## 5. Detect objects

- **Capture Photo** — takes a single snapshot, runs detection once, and
  draws labeled boxes on it.
- **Live Detection** — runs detection continuously (each frame, roughly
  30 times/second) until you tap it again to stop. Best used with the
  phone propped up rather than handheld, since detection quality depends
  on how steady/well-lit the frame is.
- **Switch Camera** — toggles between front and rear cameras.
- **Change Model** — cycles through the available YOLO model variants
  (different accuracy/speed/resolution tradeoffs). Switching models
  re-triggers the loading step in §3.
- **Reset** — clears the current overlay and stops live detection.

Below the buttons you'll see live inference time, total time (including
camera/draw overhead), and FPS stats.

## 6. Get notified when it sees a person or car

The app can push a phone notification (with the captured photo attached)
whenever it detects a **person** or **car**, rate-limited to once every
30 seconds so it doesn't spam you. This happens automatically in both
modes once you're running Live Detection or Capture Photo — you don't
need to do anything in the app itself. To actually *receive* the
notification, though, you need to subscribe once:

1. Install the **ntfy** app
   ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) /
   [iOS](https://apps.apple.com/us/app/ntfy/id1625396347)), or use
   [the web UI](https://ntfy.sh/app) in a browser.
2. Add a subscription with:
   - **Server:** `https://taco.tail9f615d.ts.net`
   - **Username:** `subscriber`
   - **Password:** the read-only token (ask whoever set up the server —
     stored in this repo's local `.env`, which is gitignored and not
     committed)
   - **Topic:** `object-detection`

Once subscribed, any person/car detection anywhere the app is used
(anyone's phone, not just yours) will push a notification to you.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Page loads but camera view is blank/black | Camera permission was denied — check your browser's site settings and re-grant it, then reload. |
| Stuck on "Loading model..." past 5 retries | Connection is too slow/unstable for the model download. Try switching to **Server-side** mode instead (much smaller download), or move to better signal/Wi-Fi. |
| Red error message instead of the camera | Read the message — it now shows the actual underlying error (e.g. a specific HTTP status or network failure) rather than a generic failure. Tap **Retry**. |
| No notification arrives despite seeing detections | You likely haven't subscribed yet — see §6. Also check you're not within the 30-second rate-limit window of the last notification. |
| Detection boxes look randomly placed/wrong | Try **Change Model** — some model variants are more accurate than others for a given scene. |

## For local development

If you're running `npm run dev` on a laptop instead of using one of the
hosted URLs above, remember phone browsers require HTTPS for camera
access (`localhost` is exempt, but a phone visiting your laptop's LAN IP
is not). Either deploy somewhere with real HTTPS, or expose your local
dev server over HTTPS yourself (e.g. `next dev --experimental-https`, or
a Tailscale Funnel like the one used for `taco.tail9f615d.ts.net`).
