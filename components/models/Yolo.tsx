import ndarray from 'ndarray';
import { Tensor } from 'onnxruntime-web';
import ops from 'ndarray-ops';
import ObjectDetectionCamera from '../ObjectDetectionCamera';
import { round } from 'lodash';
import { yoloClasses } from '../../data/yolo_classes';
import { useState } from 'react';
import { useEffect } from 'react';
import { runModelUtils } from '../../utils';
import { useNotifyDetection } from '../../utils/notify';

const RES_TO_MODEL: [number[], string][] = [
  [[256, 256], 'yolo12n.onnx'],
  [[256, 256], 'yolo11n.onnx'],
  [[256, 256], 'yolov10n.onnx'],
  [[256, 256], 'yolov7-tiny_256x256.onnx'],
  [[320, 320], 'yolov7-tiny_320x320.onnx'],
  [[640, 640], 'yolov7-tiny_640x640.onnx'],
];

// taco is served over a Tailscale Funnel, which relays all traffic through
// Tailscale's infrastructure rather than a direct connection - measured as
// low as ~15KB/s for large file transfers. A 10-25MB model/wasm file can
// take minutes, and a mid-transfer hiccup throws immediately (e.g.
// "Content-Length header of network response exceeds response Body"), so a
// generous per-attempt timeout plus automatic retry-with-backoff is needed
// rather than a single short-timeout attempt.
const SESSION_LOAD_TIMEOUT_MS = 120_000;
const MAX_LOAD_ATTEMPTS = 5;
const retryBackoffMs = (attempt: number) =>
  Math.min(30_000, 3_000 * 2 ** (attempt - 1)); // 3s, 6s, 12s, 24s, 30s...

const extractErrorDetail = (e: unknown): string =>
  // onnxruntime-web/wasm sometimes throws non-Error values (plain strings,
  // WASM abort objects) that don't survive `instanceof Error`, so fall back
  // to pulling whatever detail is available.
  e instanceof Error
    ? e.message
    : typeof e === 'string'
    ? e
    : e && typeof e === 'object' && 'message' in e
    ? String((e as { message: unknown }).message)
    : JSON.stringify(e);

const Yolo = (props: any) => {
  const [modelResolution, setModelResolution] = useState<number[]>(
    RES_TO_MODEL[0][0]
  );
  const [modelName, setModelName] = useState<string>(RES_TO_MODEL[0][1]);
  const [session, setSession] = useState<any>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [secondsUntilTimeout, setSecondsUntilTimeout] = useState<number>(
    SESSION_LOAD_TIMEOUT_MS / 1000
  );
  const [loadAttempt, setLoadAttempt] = useState(1);
  const [retryingIn, setRetryingIn] = useState<number | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const notifyDetection = useNotifyDetection();

  useEffect(() => {
    let cancelled = false;
    setSession(null);
    setSessionError(null);
    setLoadAttempt(1);
    setRetryingIn(null);

    const attemptLoad = async (attempt: number): Promise<void> => {
      if (cancelled) return;
      setLoadAttempt(attempt);
      setRetryingIn(null);
      setSecondsUntilTimeout(SESSION_LOAD_TIMEOUT_MS / 1000);

      const countdown = setInterval(() => {
        setSecondsUntilTimeout((s) => Math.max(0, s - 1));
      }, 1000);

      let settled = false;
      const onFailure = async (e: unknown) => {
        if (settled || cancelled) return;
        settled = true;
        clearInterval(countdown);
        console.error(`Model load attempt ${attempt} failed`, e);

        if (attempt >= MAX_LOAD_ATTEMPTS) {
          setSessionError(
            `Failed to load ${modelName} after ${MAX_LOAD_ATTEMPTS} attempts: ${extractErrorDetail(
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
        attemptLoad(attempt + 1);
      };

      const timeout = setTimeout(() => {
        onFailure(new Error(`Timed out after ${SESSION_LOAD_TIMEOUT_MS / 1000}s`));
      }, SESSION_LOAD_TIMEOUT_MS);

      try {
        const session = await runModelUtils.createModelCpu(
          `/runtime/${modelName}`
        );
        if (settled || cancelled) return;
        settled = true;
        clearTimeout(timeout);
        clearInterval(countdown);
        setSession(session);
      } catch (e) {
        clearTimeout(timeout);
        onFailure(e);
      }
    };

    attemptLoad(1);

    return () => {
      cancelled = true;
    };
  }, [modelName, modelResolution, retryCount]);

  const retrySessionLoad = () => setRetryCount((n) => n + 1);

  const changeModelResolution = (width?: number, height?: number) => {
    if (width !== undefined && height !== undefined) {
      setModelResolution([width, height]);
      return;
    }
    const index = RES_TO_MODEL.findIndex((item) => item[0] === modelResolution);
    if (index === RES_TO_MODEL.length - 1) {
      setModelResolution(RES_TO_MODEL[0][0]);
      setModelName(RES_TO_MODEL[0][1]);
    } else {
      setModelResolution(RES_TO_MODEL[index + 1][0]);
      setModelName(RES_TO_MODEL[index + 1][1]);
    }
  };

  const resizeCanvasCtx = (
    ctx: CanvasRenderingContext2D,
    targetWidth: number,
    targetHeight: number,
    inPlace = false
  ) => {
    let canvas: HTMLCanvasElement;

    if (inPlace) {
      // Get the canvas element that the context is associated with
      canvas = ctx.canvas;

      // Set the canvas dimensions to the target width and height
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Scale the context to the new dimensions
      ctx.scale(
        targetWidth / canvas.clientWidth,
        targetHeight / canvas.clientHeight
      );
    } else {
      // Create a new canvas element with the target dimensions
      canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Draw the source canvas into the target canvas
      canvas
        .getContext('2d')!
        .drawImage(ctx.canvas, 0, 0, targetWidth, targetHeight);

      // Get a new rendering context for the new canvas
      ctx = canvas.getContext('2d')!;
    }

    return ctx;
  };

  const preprocess = (ctx: CanvasRenderingContext2D) => {
    const resizedCtx = resizeCanvasCtx(
      ctx,
      modelResolution[0],
      modelResolution[1]
    );

    const imageData = resizedCtx.getImageData(
      0,
      0,
      modelResolution[0],
      modelResolution[1]
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    ops.divseq(dataProcessedTensor, 255);

    const tensor = new Tensor('float32', new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  };

  const conf2color = (conf: number) => {
    const r = Math.round(255 * (1 - conf));
    const g = Math.round(255 * conf);
    return `rgb(${r},${g},0)`;
  };

  // map model name to corresponding postprocess function
  const postprocessMap: Record<string, PostprocessFunction> = {
    'yolo12n.onnx': postprocessYolov10,
    'yolo11n.onnx': postprocessYolov10,
    'yolov10n.onnx': postprocessYolov10,
    'yolov7-tiny_256x256.onnx': postprocessYolov7,
    'yolov7-tiny_320x320.onnx': postprocessYolov7,
    'yolov7-tiny_640x640.onnx': postprocessYolov7,
  };

  const postprocess = async (
    tensor: Tensor,
    inferenceTime: number,
    ctx: CanvasRenderingContext2D,
    modelName: string
  ) => {
    // Output tensor of yolov7-tiny is [det_num, 7]
    // while yolov10n/yolo11n/yolo12n are [1, 300, 6] - the yolo11/12 .onnx
    // files are exported with NMS baked into the graph (`nms=True`, see
    // convert_pt_to_onnx/ultralytics_pt_to_onnx.md), which happens to
    // produce the exact same [x0,y0,x1,y1,score,cls_id] layout yolov10n's
    // export already used, pre-sorted descending and zero-padded to 300
    // rows - so postprocessYolov10 handles all three with no changes.
    // Thus we need to handle them differently

    if (modelName in postprocessMap) {
      console.log('Using postprocess for', modelName);
      postprocessMap[modelName](ctx, modelResolution, tensor, conf2color, (detectedClasses) =>
        notifyDetection(ctx, detectedClasses)
      );
    }
  };

  const detect = async (ctx: CanvasRenderingContext2D): Promise<number> => {
    const data = preprocess(ctx);
    const [outputTensor, inferenceTime] = await runModelUtils.runModel(
      session,
      data
    );
    await postprocess(outputTensor, inferenceTime, ctx, modelName);
    return inferenceTime;
  };

  return (
    <ObjectDetectionCamera
      width={props.width}
      height={props.height}
      ready={!!session}
      detect={detect}
      sessionError={sessionError}
      secondsUntilTimeout={secondsUntilTimeout}
      loadAttempt={loadAttempt}
      maxLoadAttempts={MAX_LOAD_ATTEMPTS}
      retryingIn={retryingIn}
      onRetrySession={retrySessionLoad}
      changeCurrentModelResolution={changeModelResolution}
      currentModelResolution={modelResolution}
      modelName={modelName}
    />
  );
};

export default Yolo;

type PostprocessFunction = (
  ctx: CanvasRenderingContext2D,
  modelResolution: number[],
  tensor: Tensor,
  conf2color: (conf: number) => string,
  onDetections: (detectedClasses: string[]) => void
) => void;

const postprocessYolov10: PostprocessFunction = (
  ctx: CanvasRenderingContext2D,
  modelResolution: number[],
  tensor: Tensor,
  conf2color: (conf: number) => string,
  onDetections: (detectedClasses: string[]) => void
) => {
  const dx = ctx.canvas.width / modelResolution[0];
  const dy = ctx.canvas.height / modelResolution[1];

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const detectedClasses: string[] = [];
  let x0, y0, x1, y1, cls_id, score;
  // yolov10n output tensor is [1, all_boxes, 6]
  // console.log(tensor.dims);
  for (let i = 0; i < tensor.dims[1]; i += 6) {
    [x0, y0, x1, y1, score, cls_id] = tensor.data.slice(i, i + 6);
    if ((score as any) < 0.25) {
      break;
    }
    detectedClasses.push(yoloClasses[cls_id as any]);

    // scale to canvas size
    [x0, x1] = [x0, x1].map((x: any) => x * dx);
    [y0, y1] = [y0, y1].map((x: any) => x * dy);

    [x0, y0, x1, y1, cls_id] = [x0, y0, x1, y1, cls_id].map((x: any) =>
      round(x)
    );

    [score] = [score].map((x: any) => round(x * 100, 1));
    const label =
      yoloClasses[cls_id].toString()[0].toUpperCase() +
      yoloClasses[cls_id].toString().substring(1) +
      ' ' +
      score.toString() +
      '%';
    const color = conf2color(score / 100);

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.font = '20px Arial';
    ctx.fillStyle = color;
    ctx.fillText(label, x0, y0 - 5);

    // fillrect with transparent color
    ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }

  onDetections(detectedClasses);
};

const postprocessYolov7: PostprocessFunction = (
  ctx: CanvasRenderingContext2D,
  modelResolution: number[],
  tensor: Tensor,
  conf2color: (conf: number) => string,
  onDetections: (detectedClasses: string[]) => void
) => {
  const dx = ctx.canvas.width / modelResolution[0];
  const dy = ctx.canvas.height / modelResolution[1];

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const detectedClasses: string[] = [];
  let batch_id, x0, y0, x1, y1, cls_id, score;
  // Output tensor of yolov7-tiny is [det_num, 7]
  // console.log(tensor.dims);
  for (let i = 0; i < tensor.dims[0]; i++) {
    [batch_id, x0, y0, x1, y1, cls_id, score] = tensor.data.slice(
      i * 7,
      i * 7 + 7
    );
    detectedClasses.push(yoloClasses[cls_id as any]);

    // scale to canvas size
    [x0, x1] = [x0, x1].map((x: any) => x * dx);
    [y0, y1] = [y0, y1].map((x: any) => x * dy);

    [x0, y0, x1, y1, cls_id] = [x0, y0, x1, y1, cls_id].map((x: any) =>
      round(x)
    );

    [score] = [score].map((x: any) => round(x * 100, 1));
    const label =
      yoloClasses[cls_id].toString()[0].toUpperCase() +
      yoloClasses[cls_id].toString().substring(1) +
      ' ' +
      score.toString() +
      '%';
    const color = conf2color(score / 100);

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.font = '20px Arial';
    ctx.fillStyle = color;
    ctx.fillText(label, x0, y0 - 5);

    // fillrect with transparent color
    ctx.fillStyle = color.replace(')', ', 0.2)').replace('rgb', 'rgba');
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }

  onDetections(detectedClasses);
};
