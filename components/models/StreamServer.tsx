import ObjectDetectionCamera from '../ObjectDetectionCamera';
import { useRef, useState, useEffect } from 'react';
import { useNotifyDetection } from '../../utils/notify';
import { INFERENCE_ENDPOINT } from '../../utils/inferenceEndpoint';
import { Detection } from '../../utils/detectionTypes';

// Like YoloServer.tsx, but captures full native camera resolution (instead
// of the on-page display size) and self-paces to whatever frame rate the
// upload + server round trip can actually sustain, rather than firing a
// request as fast as the live-detection loop allows. Full-resolution JPEGs
// are much bigger than the small display-resolution frames the other
// server modes send, so a fixed interval (like AlprServer.tsx's 1s) would
// either be too conservative on a fast link or still overload a slow one -
// adaptive pacing tracks the actual measured round trip instead of
// guessing a fixed number up front.

const MODELS = [
  'yolo12n.onnx',
  'yolo11n.onnx',
  'yolov10n.onnx',
  'yolov7-tiny_256x256.onnx',
  'yolov7-tiny_320x320.onnx',
  'yolov7-tiny_640x640.onnx',
];

const HEALTH_CHECK_TIMEOUT_MS = 10_000;
const MAX_HEALTH_CHECK_ATTEMPTS = 5;
const retryBackoffMs = (attempt: number) =>
  Math.min(15_000, 2_000 * 2 ** (attempt - 1));

// The 1-6 fps band this mode is allowed to settle into. Below 1fps isn't
// worth calling "streaming" any more; above 6fps there's no benefit to a
// full-resolution frame (a small display-resolution frame would do, see
// YoloServer.tsx) and it risks flooding taco with large uploads.
const MIN_FPS = 1;
const MAX_FPS = 6;
const MAX_INTERVAL_MS = 1000 / MIN_FPS;
const MIN_INTERVAL_MS = 1000 / MAX_FPS;
// Exponential-moving-average weight for folding each new round-trip
// measurement into the pacing interval - smooths out one-off network
// blips without ignoring a sustained speed change for too long.
const INTERVAL_SMOOTHING = 0.5;

const extractErrorDetail = (e: unknown): string =>
  e instanceof Error
    ? e.message
    : typeof e === 'string'
    ? e
    : e && typeof e === 'object' && 'message' in e
    ? String((e as { message: unknown }).message)
    : JSON.stringify(e);

const StreamServer = (props: any) => {
  const [modelIndex, setModelIndex] = useState(0);
  const modelName = MODELS[modelIndex];
  const [ready, setReady] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [secondsUntilTimeout, setSecondsUntilTimeout] = useState(
    HEALTH_CHECK_TIMEOUT_MS / 1000
  );
  const [loadAttempt, setLoadAttempt] = useState(1);
  const [retryingIn, setRetryingIn] = useState<number | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const notifyDetection = useNotifyDetection();

  // Starting guess before any round trip has been measured - refined
  // toward whatever the link/server can actually sustain after the first
  // frame (see the EMA update in detect() below).
  const intervalMsRef = useRef((MIN_INTERVAL_MS + MAX_INTERVAL_MS) / 2);
  const lastSendAtRef = useRef(0);
  const [currentFps, setCurrentFps] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setSessionError(null);
    setLoadAttempt(1);
    setRetryingIn(null);

    const attemptCheck = async (attempt: number): Promise<void> => {
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
        console.error(`Inference server health check attempt ${attempt} failed`, e);

        if (attempt >= MAX_HEALTH_CHECK_ATTEMPTS) {
          setSessionError(
            `Inference server unreachable after ${MAX_HEALTH_CHECK_ATTEMPTS} attempts: ${extractErrorDetail(
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
        attemptCheck(attempt + 1);
      };

      const controller = new AbortController();
      const timeout = setTimeout(() => {
        controller.abort();
        onFailure(new Error(`Timed out after ${HEALTH_CHECK_TIMEOUT_MS / 1000}s`));
      }, HEALTH_CHECK_TIMEOUT_MS);

      try {
        const res = await fetch(`${INFERENCE_ENDPOINT}/health`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Health check returned HTTP ${res.status}`);
        if (settled || cancelled) return;
        settled = true;
        clearTimeout(timeout);
        clearInterval(countdown);
        setReady(true);
      } catch (e) {
        clearTimeout(timeout);
        onFailure(e);
      }
    };

    attemptCheck(1);

    return () => {
      cancelled = true;
    };
  }, [retryCount]);

  const retrySessionLoad = () => setRetryCount((n) => n + 1);

  const changeModel = () => {
    setModelIndex((i) => (i + 1) % MODELS.length);
  };

  const conf2color = (conf: number) => {
    const r = Math.round(255 * (1 - conf));
    const g = Math.round(255 * conf);
    return `rgb(${r},${g},0)`;
  };

  const detect = async (ctx: CanvasRenderingContext2D): Promise<number> => {
    const sinceLastSend = Date.now() - lastSendAtRef.current;
    if (sinceLastSend < intervalMsRef.current) {
      await new Promise((resolve) =>
        setTimeout(resolve, intervalMsRef.current - sinceLastSend)
      );
    }

    const blob: Blob | null = await new Promise((resolve) =>
      ctx.canvas.toBlob(resolve, 'image/jpeg', 0.9)
    );
    if (!blob) return 0;

    const requestStart = Date.now();
    lastSendAtRef.current = requestStart;

    const res = await fetch(
      `${INFERENCE_ENDPOINT}/predict?model=${encodeURIComponent(modelName)}`,
      { method: 'POST', body: blob }
    );
    if (!res.ok) {
      throw new Error(`Inference request failed: HTTP ${res.status}`);
    }
    const result: { inferenceTimeMs: number; detections: Detection[] } =
      await res.json();

    // Pace future frames to whatever this round trip (upload + inference +
    // download, not just the server's own inferenceTimeMs) actually took,
    // smoothed and clamped to the 1-6fps band.
    const roundTripMs = Date.now() - requestStart;
    intervalMsRef.current = Math.min(
      MAX_INTERVAL_MS,
      Math.max(
        MIN_INTERVAL_MS,
        INTERVAL_SMOOTHING * roundTripMs +
          (1 - INTERVAL_SMOOTHING) * intervalMsRef.current
      )
    );
    setCurrentFps(Math.round((1000 / intervalMsRef.current) * 10) / 10);

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const detectedClasses: string[] = [];
    for (const d of result.detections) {
      detectedClasses.push(d.class);
      const [nx0, ny0, nx1, ny1] = d.box;
      const x0 = nx0 * ctx.canvas.width;
      const y0 = ny0 * ctx.canvas.height;
      const x1 = nx1 * ctx.canvas.width;
      const y1 = ny1 * ctx.canvas.height;

      const score = Math.round(d.confidence * 100 * 10) / 10;
      const label =
        d.class[0].toUpperCase() + d.class.substring(1) + ' ' + score + '%';
      const color = conf2color(d.confidence);

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.font = '20px Arial';
      ctx.fillStyle = color;
      ctx.fillText(label, x0, y0 - 5);
      ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
    notifyDetection(ctx, detectedClasses);

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
      changeCurrentModelResolution={changeModel}
      currentModelResolution={[0, 0]}
      nativeResolutionCapture={true}
      modelName={`${modelName} - full-res stream${
        currentFps ? ` (~${currentFps} fps)` : ' (1-6 fps, adaptive)'
      }`}
    />
  );
};

export default StreamServer;
