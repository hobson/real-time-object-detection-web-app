import ObjectDetectionCamera from '../ObjectDetectionCamera';
import { useEffect, useRef, useState } from 'react';

// License plate detection + OCR, server-side only (there is no in-browser
// equivalent of Yolo.tsx for this - the OCR model makes a WASM download
// even less attractive than for the COCO detector, see
// docs/PLAN-realtime-license-plate-detection.md §4).
//
// Streams frames to /inference-server's /alpr/ws over a single persistent
// WebSocket rather than POSTing each frame like YoloServer.tsx does for
// the COCO models. At the ~1 fps this endpoint is designed for, a video
// codec (WebRTC) buys nothing - every frame is effectively a keyframe at
// that rate - and a plain per-frame POST would re-pay a request/response
// round trip on every frame. A single WebSocket amortizes that setup cost
// once for the whole session and lets the server push results back the
// moment they're ready. See inference-server/alpr.py's module docstring
// for the full reasoning.

const INFERENCE_ENDPOINT =
  process.env.NEXT_PUBLIC_INFERENCE_URL ||
  'https://taco.tail9f615d.ts.net:8443/infer';

const WS_ENDPOINT = INFERENCE_ENDPOINT.replace(/^http/, 'ws') + '/alpr/ws';

const HEALTH_CHECK_TIMEOUT_MS = 10_000;
const MAX_HEALTH_CHECK_ATTEMPTS = 5;
const retryBackoffMs = (attempt: number) =>
  Math.min(15_000, 2_000 * 2 ** (attempt - 1));

// This endpoint is designed to be driven at ~1 fps: the OCR pass is more
// expensive than plain detection, and there's no benefit to reading the
// same plate faster than once a second. Enforced client-side (the server
// processes whatever it's sent, in order, with no rate limiting of its
// own - see alpr.py) so this is the single place the cadence is decided.
const MIN_FRAME_INTERVAL_MS = 1000;

const extractErrorDetail = (e: unknown): string =>
  e instanceof Error
    ? e.message
    : typeof e === 'string'
    ? e
    : e && typeof e === 'object' && 'message' in e
    ? String((e as { message: unknown }).message)
    : JSON.stringify(e);

type PlateDetection = {
  box: number[];
  detectionConfidence: number;
  plate: string | null;
  ocrConfidence: number | null;
  region: string | null;
  regionConfidence: number | null;
};

type AlprResponse = {
  inferenceTimeMs: number;
  detections: PlateDetection[];
  error?: string;
};

const AlprServer = (props: any) => {
  const [ready, setReady] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [secondsUntilTimeout, setSecondsUntilTimeout] = useState(
    HEALTH_CHECK_TIMEOUT_MS / 1000
  );
  const [loadAttempt, setLoadAttempt] = useState(1);
  const [retryingIn, setRetryingIn] = useState<number | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  // Requests are answered in the order they're sent (the server has a
  // single sequential receive loop - see alpr.py), so a FIFO queue of
  // pending resolvers is enough to correlate each `detect()` call with
  // its response without needing per-message request IDs.
  const pendingRef = useRef<
    Array<{ resolve: (r: AlprResponse) => void; reject: (e: Error) => void }>
  >([]);
  const lastSendAtRef = useRef<number>(0);

  // Opens the WebSocket once models are confirmed loaded server-side
  // (GET /alpr/health), with the same retry-with-backoff shape as
  // YoloServer.tsx's health check.
  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    setReady(false);
    setSessionError(null);
    setLoadAttempt(1);
    setRetryingIn(null);

    const attemptConnect = async (attempt: number): Promise<void> => {
      if (cancelled) return;
      setLoadAttempt(attempt);
      setRetryingIn(null);
      setSecondsUntilTimeout(HEALTH_CHECK_TIMEOUT_MS / 1000);

      const countdown = setInterval(() => {
        setSecondsUntilTimeout((s) => Math.max(0, s - 1));
      }, 1000);

      let settled = false;
      const onFailure = async (e: unknown) => {
        if (settled || cancelled) return;
        settled = true;
        clearInterval(countdown);
        console.error(`ALPR server connection attempt ${attempt} failed`, e);

        if (attempt >= MAX_HEALTH_CHECK_ATTEMPTS) {
          setSessionError(
            `ALPR server unreachable after ${MAX_HEALTH_CHECK_ATTEMPTS} attempts: ${extractErrorDetail(
              e
            )}`
          );
          return;
        }

        const backoffMs = retryBackoffMs(attempt);
        let remaining = Math.ceil(backoffMs / 1000);
        setRetryingIn(remaining);
        const backoffInterval = setInterval(() => {
          remaining -= 1;
          setRetryingIn(Math.max(0, remaining));
        }, 1000);
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
        clearInterval(backoffInterval);
        if (cancelled) return;
        attemptConnect(attempt + 1);
      };

      const controller = new AbortController();
      const timeout = setTimeout(() => {
        controller.abort();
        onFailure(new Error(`Timed out after ${HEALTH_CHECK_TIMEOUT_MS / 1000}s`));
      }, HEALTH_CHECK_TIMEOUT_MS);

      try {
        const res = await fetch(`${INFERENCE_ENDPOINT}/alpr/health`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Health check returned HTTP ${res.status}`);
        if (settled || cancelled) return;

        ws = new WebSocket(WS_ENDPOINT);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
          if (settled || cancelled) return;
          settled = true;
          clearTimeout(timeout);
          clearInterval(countdown);
          wsRef.current = ws;
          setReady(true);
        };
        ws.onmessage = (ev) => {
          const pending = pendingRef.current.shift();
          if (!pending) return; // stray message, nothing waiting on it
          try {
            const parsed: AlprResponse = JSON.parse(ev.data);
            if (parsed.error) {
              pending.reject(new Error(parsed.error));
            } else {
              pending.resolve(parsed);
            }
          } catch (e) {
            pending.reject(e instanceof Error ? e : new Error(String(e)));
          }
        };
        ws.onerror = () => {
          if (!settled) onFailure(new Error('WebSocket error'));
        };
        ws.onclose = () => {
          // Fail any in-flight detect() calls so they don't hang forever.
          pendingRef.current.splice(0).forEach((p) =>
            p.reject(new Error('ALPR WebSocket closed'))
          );
          if (!cancelled) setReady(false);
          if (!settled) onFailure(new Error('WebSocket closed before opening'));
        };
      } catch (e) {
        clearTimeout(timeout);
        onFailure(e);
      }
    };

    attemptConnect(1);

    return () => {
      cancelled = true;
      pendingRef.current.splice(0).forEach((p) =>
        p.reject(new Error('Component unmounted'))
      );
      ws?.close();
    };
  }, [retryCount]);

  const retrySessionLoad = () => setRetryCount((n) => n + 1);

  const sendFrame = (blob: Blob): Promise<AlprResponse> => {
    return new Promise((resolve, reject) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('ALPR WebSocket not connected'));
        return;
      }
      pendingRef.current.push({ resolve, reject });
      blob
        .arrayBuffer()
        .then((buf) => ws.send(buf))
        .catch((e) => reject(e instanceof Error ? e : new Error(String(e))));
    });
  };

  const detect = async (ctx: CanvasRenderingContext2D): Promise<number> => {
    const elapsed = Date.now() - lastSendAtRef.current;
    if (elapsed < MIN_FRAME_INTERVAL_MS) {
      await new Promise((resolve) =>
        setTimeout(resolve, MIN_FRAME_INTERVAL_MS - elapsed)
      );
    }
    lastSendAtRef.current = Date.now();

    const blob: Blob | null = await new Promise((resolve) =>
      ctx.canvas.toBlob(resolve, 'image/jpeg', 0.85)
    );
    if (!blob) return 0;

    const result = await sendFrame(blob);

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (const d of result.detections) {
      const [nx0, ny0, nx1, ny1] = d.box;
      const x0 = nx0 * ctx.canvas.width;
      const y0 = ny0 * ctx.canvas.height;
      const x1 = nx1 * ctx.canvas.width;
      const y1 = ny1 * ctx.canvas.height;

      const color = 'rgb(56,189,248)'; // sky-400, distinct from the COCO boxes' green/red scale
      const label = d.plate
        ? `${d.plate}${d.region ? ` (${d.region})` : ''} ${Math.round(
            (d.ocrConfidence ?? 0) * 100
          )}%`
        : `plate ${Math.round(d.detectionConfidence * 100)}%`;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.font = '20px Arial';
      ctx.fillStyle = color;
      ctx.fillText(label, x0, y0 - 5);
      ctx.fillStyle = 'rgba(56,189,248,0.2)';
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }

    return result.inferenceTimeMs;
  };

  return (
    <ObjectDetectionCamera
      width={props.width}
      height={props.height}
      ready={ready}
      detect={detect}
      sessionError={sessionError}
      secondsUntilTimeout={secondsUntilTimeout}
      loadAttempt={loadAttempt}
      maxLoadAttempts={MAX_HEALTH_CHECK_ATTEMPTS}
      retryingIn={retryingIn}
      onRetrySession={retrySessionLoad}
      changeCurrentModelResolution={() => {}}
      currentModelResolution={[0, 0]}
      modelName="fast-alpr (server, ~1 fps)"
    />
  );
};

export default AlprServer;
