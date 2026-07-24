import ObjectDetectionCamera from '../ObjectDetectionCamera';
import { useState, useRef, useEffect } from 'react';

// Alternative to Yolo.tsx: instead of downloading a ~20MB wasm runtime +
// model and running inference in the browser, capture a frame and post it
// to a small inference server (see /inference-server) that runs the same
// ONNX models with the same postprocessing, server-side. Trades a one-time
// large download for a small recurring per-frame upload - see CLAUDE.md
// ("Server-side inference") for the tradeoffs that motivated this.

const MODELS = [
  'yolo12n.onnx',
  'yolo11n.onnx',
  'yolov10n.onnx',
  'yolov7-tiny_256x256.onnx',
  'yolov7-tiny_320x320.onnx',
  'yolov7-tiny_640x640.onnx',
];

const INFERENCE_ENDPOINT =
  process.env.NEXT_PUBLIC_INFERENCE_URL ||
  'https://taco.tail9f615d.ts.net:8443/infer';

const NOTIFY_CLASSES = new Set(['person', 'car']);
const NOTIFY_MIN_INTERVAL_MS = 30_000;
const NOTIFY_ENDPOINT =
  process.env.NEXT_PUBLIC_NOTIFY_URL ||
  'https://taco.tail9f615d.ts.net:8443/notify';

// The inference server itself is small and fast to reach (no large
// download), so a much shorter timeout/retry budget than the client-side
// model-loading path is appropriate here.
const HEALTH_CHECK_TIMEOUT_MS = 10_000;
const MAX_HEALTH_CHECK_ATTEMPTS = 5;
const retryBackoffMs = (attempt: number) =>
  Math.min(15_000, 2_000 * 2 ** (attempt - 1));

const extractErrorDetail = (e: unknown): string =>
  e instanceof Error
    ? e.message
    : typeof e === 'string'
    ? e
    : e && typeof e === 'object' && 'message' in e
    ? String((e as { message: unknown }).message)
    : JSON.stringify(e);

type Detection = { class: string; confidence: number; box: number[] };

const YoloServer = (props: any) => {
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
  const lastNotifiedAt = useRef<number>(0);

  // Confirms the inference server is actually reachable before enabling
  // the capture/live-detection buttons, with the same retry-with-backoff
  // shape as the client-side model loader (just much shorter budgets).
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

  const notifyDetection = (ctx: CanvasRenderingContext2D, classes: string[]) => {
    const matched = classes.filter((c) => NOTIFY_CLASSES.has(c));
    if (matched.length === 0) return;

    const now = Date.now();
    if (now - lastNotifiedAt.current < NOTIFY_MIN_INTERVAL_MS) return;
    lastNotifiedAt.current = now;

    ctx.canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const uniqueClasses = Array.from(new Set(matched));
        fetch(
          `${NOTIFY_ENDPOINT}?title=${encodeURIComponent(
            uniqueClasses.join(', ') + ' detected'
          )}&tags=camera`,
          { method: 'POST', body: blob }
        ).catch((e) => console.error('Failed to send ntfy notification', e));
      },
      'image/jpeg',
      0.8
    );
  };

  const detect = async (ctx: CanvasRenderingContext2D): Promise<number> => {
    const blob: Blob | null = await new Promise((resolve) =>
      ctx.canvas.toBlob(resolve, 'image/jpeg', 0.85)
    );
    if (!blob) return 0;

    const res = await fetch(
      `${INFERENCE_ENDPOINT}/predict?model=${encodeURIComponent(modelName)}`,
      { method: 'POST', body: blob }
    );
    if (!res.ok) {
      throw new Error(`Inference request failed: HTTP ${res.status}`);
    }
    const result: { inferenceTimeMs: number; detections: Detection[] } =
      await res.json();

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
      modelName={modelName}
    />
  );
};

export default YoloServer;
