"""Manual end-to-end test client for the /alpr/ws streaming endpoint.

Captures frames from a webcam, a video file, an animated GIF, a directory
of still images, or a single still image (sent on a loop), streams them to
a running inference-server over the same WebSocket protocol
`components/models/AlprServer.tsx` uses, and displays the annotated result
(bounding box + plate text/region) either in a matplotlib window or in a
web browser (as an MJPEG stream). Optionally saves each annotated frame to
disk and assembles them into a GIF for offline review, and prints a
throughput summary (frames/sec, mean/median/p95 latency).

This does not start the server itself - run it separately first:

    uvicorn main:app --reload

Usage:

    python test_alpr_stream.py --source webcam
    python test_alpr_stream.py --source path/to/video.mp4 --fps 1
    python test_alpr_stream.py --source ../../fast-alpr/assets/test_image.png --display browser
    python test_alpr_stream.py --url ws://taco.tail9f615d.ts.net:8443/infer/alpr/ws

    # Throughput test: send as fast as the server can keep up (no client-side
    # pacing) and measure it, saving an annotated GIF for a human to review:
    python test_alpr_stream.py --source ../../fast-alpr/assets/alpr.gif \\
        --fps 1000 --max-frames 22 --display none --save-dir /tmp/alpr_check

    # Accuracy spot-check across many distinct real plates/scenes:
    python test_alpr_stream.py --source ../../data/license_plates/images \\
        --max-frames 20 --display none --save-dir /tmp/alpr_accuracy

Extra deps beyond requirements.txt (server-only, so kept out of it):
    pip install -r requirements-dev.txt
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import socketserver
import statistics
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import websockets
from PIL import Image, ImageSequence

DEFAULT_WS_URL = "ws://localhost:8000/alpr/ws"
BOX_COLOR = (56, 189, 248)  # BGR, matches AlprServer.tsx's sky-400 boxes
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    detection_confidence: float
    plate: str | None
    ocr_confidence: float | None
    region: str | None
    region_confidence: float | None


def _load_gif_frames(path: str) -> list[np.ndarray]:
    im = Image.open(path)
    frames = [
        cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        for frame in ImageSequence.Iterator(im)
    ]
    if not frames:
        raise RuntimeError(f"'{path}' has no frames")
    return frames


@dataclass
class FrameSource:
    """Yields BGR frames from a webcam, video file, animated GIF, a directory
    of still images (cycled in sorted order), or a single looped image."""

    path: str
    fps: float
    _cap: cv2.VideoCapture | None = field(default=None, init=False, repr=False)
    _frames: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path == "webcam":
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError("Could not open webcam (device 0)")
            return

        p = Path(self.path)

        if p.is_dir():
            image_paths = sorted(
                f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise RuntimeError(f"No images found in directory '{self.path}'")
            frames = [cv2.imread(str(f)) for f in image_paths]
            missing = [f for f, im in zip(image_paths, frames) if im is None]
            if missing:
                raise RuntimeError(f"Could not decode: {missing}")
            self._frames = frames
            return

        if p.suffix.lower() == ".gif":
            self._frames = _load_gif_frames(self.path)
            return

        still = cv2.imread(self.path)
        if still is not None:
            self._frames = [still]
            return

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open '{self.path}' as an image, GIF, directory, or video")

    def read(self) -> np.ndarray:
        if self._frames is not None:
            frame = self._frames[self._index % len(self._frames)].copy()
            self._index += 1
            return frame
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            # Video file exhausted - loop back to the start rather than stopping,
            # so a short test clip still exercises a long-running stream test.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from '{self.path}'")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def _parse_detections(payload: dict) -> list[Detection]:
    detections = []
    for d in payload.get("detections", []):
        detections.append(
            Detection(
                box=tuple(d["box"]),
                detection_confidence=d["detectionConfidence"],
                plate=d.get("plate"),
                ocr_confidence=d.get("ocrConfidence"),
                region=d.get("region"),
                region_confidence=d.get("regionConfidence"),
            )
        )
    return detections


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for d in detections:
        x0, y0, x1, y1 = d.box
        p0 = (int(x0 * width), int(y0 * height))
        p1 = (int(x1 * width), int(y1 * height))
        cv2.rectangle(annotated, p0, p1, BOX_COLOR, 2)

        if d.plate:
            label = f"{d.plate}"
            if d.region:
                label += f" ({d.region})"
            label += f" {round((d.ocr_confidence or 0) * 100)}%"
        else:
            label = f"plate {round(d.detection_confidence * 100)}%"

        text_y = max(p0[1] - 8, 15)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (p0[0], text_y - th - 4), (p0[0] + tw, text_y + 4), BOX_COLOR, -1)
        cv2.putText(
            annotated, label, (p0[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )
    return annotated


class NullDisplay:
    """No live display - use with --save-dir for a pure throughput/accuracy run."""

    closed = False

    def show(self, frame_bgr: np.ndarray, status: str) -> None:
        pass

    def close(self) -> None:
        pass


class MatplotlibDisplay:
    """Live-updating matplotlib window, refreshed once per received frame."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.set_title("fast-alpr streaming test (close window to stop)")
        self._ax.axis("off")
        self._im = None
        self._closed = False
        self._fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def show(self, frame_bgr: np.ndarray, status: str) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._im is None:
            self._im = self._ax.imshow(rgb)
        else:
            self._im.set_data(rgb)
        self._ax.set_xlabel(status)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self._fig)


class BrowserDisplay:
    """Serves the latest annotated frame as an MJPEG stream for a browser tab."""

    def __init__(self, port: int = 8642) -> None:
        self._latest_jpeg = b""
        self._lock = threading.Lock()
        self._closed = False

        latest_jpeg_ref = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # quiet
                pass

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path != "/":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while not latest_jpeg_ref._closed:
                        with latest_jpeg_ref._lock:
                            jpeg = latest_jpeg_ref._latest_jpeg
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{port}/"
        print(f"Serving annotated stream at {url} (opening browser tab)")
        webbrowser.open(url)

    def show(self, frame_bgr: np.ndarray, status: str) -> None:
        cv2.putText(
            frame_bgr, status, (10, frame_bgr.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self._httpd.shutdown()


def _print_throughput_summary(round_trip_ms: list[float], wall_start: float) -> None:
    if not round_trip_ms:
        print("No frames processed.")
        return
    wall_elapsed = time.monotonic() - wall_start
    print("\n--- Throughput summary ---")
    print(f"Frames processed:     {len(round_trip_ms)}")
    print(f"Wall time:            {wall_elapsed:.2f}s")
    print(f"Achieved rate:        {len(round_trip_ms) / wall_elapsed:.2f} fps")
    print(f"Round-trip mean:      {statistics.mean(round_trip_ms):.1f}ms")
    print(f"Round-trip median:    {statistics.median(round_trip_ms):.1f}ms")
    if len(round_trip_ms) > 1:
        print(f"Round-trip p95:       {statistics.quantiles(round_trip_ms, n=20)[18]:.1f}ms")
    print(f"Round-trip min/max:   {min(round_trip_ms):.1f}ms / {max(round_trip_ms):.1f}ms")


async def run(args: argparse.Namespace) -> None:
    source = FrameSource(args.source, args.fps)
    display = {
        "matplotlib": MatplotlibDisplay,
        "browser": lambda: BrowserDisplay(args.port),
        "none": NullDisplay,
    }[args.display]()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    saved_frames: list[np.ndarray] = []

    round_trip_ms: list[float] = []
    wall_start = time.monotonic()

    print(f"Connecting to {args.url} ...")
    try:
        async with websockets.connect(args.url, max_size=None) as ws:
            print("Connected. Streaming at "
                  f"{args.fps} fps from '{args.source}' (Ctrl+C to stop"
                  + (f", stops after {args.max_frames} frames" if args.max_frames else "")
                  + ").")
            frame_count = 0
            try:
                while not display.closed and (
                    args.max_frames == 0 or frame_count < args.max_frames
                ):
                    loop_start = time.monotonic()
                    frame = source.read()

                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        continue

                    send_time = time.monotonic()
                    await ws.send(jpeg.tobytes())
                    reply = json.loads(await ws.recv())
                    round_trip = (time.monotonic() - send_time) * 1000
                    round_trip_ms.append(round_trip)

                    if "error" in reply:
                        print(f"Server error: {reply['error']}")
                        annotated = frame
                        status = f"error: {reply['error']}"
                    else:
                        detections = _parse_detections(reply)
                        annotated = draw_detections(frame, detections)
                        plates = ", ".join(d.plate for d in detections if d.plate) or "none"
                        status = (
                            f"frame {frame_count} | inference "
                            f"{reply['inferenceTimeMs']:.1f}ms | round-trip "
                            f"{round_trip:.1f}ms | plates: {plates}"
                        )
                        print(status)

                    display.show(annotated, status)
                    if save_dir is not None:
                        cv2.imwrite(str(save_dir / f"frame_{frame_count:04d}.jpg"), annotated)
                        saved_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    frame_count += 1

                    elapsed = time.monotonic() - loop_start
                    sleep_for = max(0.0, (1.0 / args.fps) - elapsed)
                    await asyncio.sleep(sleep_for)
            except KeyboardInterrupt:
                pass
    finally:
        source.close()
        display.close()
        _print_throughput_summary(round_trip_ms, wall_start)
        if save_dir is not None and saved_frames:
            gif_path = save_dir / "output.gif"
            pil_frames = [Image.fromarray(f) for f in saved_frames]
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=max(1, round(1000 / args.fps)), loop=0,
            )
            print(f"Saved {len(saved_frames)} annotated frames + {gif_path} to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--url", default=DEFAULT_WS_URL,
        help=f"ALPR WebSocket URL (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "--source", default="webcam",
        help="'webcam', a video file path, or a still image path (default: webcam)",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Frames per second to send - matches the endpoint's intended ~1 fps (default: 1.0)",
    )
    parser.add_argument(
        "--display", choices=["matplotlib", "browser", "none"], default="matplotlib",
        help="Where to show annotated frames (default: matplotlib)",
    )
    parser.add_argument(
        "--port", type=int, default=8642,
        help="Local port for --display browser's MJPEG server (default: 8642)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many frames (0 = run until Ctrl+C or the display window closes, default: 0)",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save annotated frame_NNNN.jpg files + an assembled output.gif into",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
