import Webcam from 'react-webcam';
import { useRef, useState, useEffect, useLayoutEffect } from 'react';

const ObjectDetectionCamera = (props: {
  width: number;
  height: number;
  modelName: string;
  // True once detection can actually run - e.g. a local WASM session
  // finished loading, or a remote inference server responded healthy.
  ready: boolean;
  sessionError: string | null;
  secondsUntilTimeout: number;
  loadAttempt: number;
  maxLoadAttempts: number;
  retryingIn: number | null;
  onRetrySession: () => void;
  // Fully owns one detection pass: capture -> infer (locally or over the
  // network) -> draw boxes on ctx -> return inference time in ms. Keeps
  // this component ignorant of *how* inference happens. `isSingleCapture`
  // is true only for the explicit "Capture Photo" button, false for every
  // frame of "Live Detection" - lets a mode treat a deliberate one-off
  // capture differently from the continuous loop (e.g. Yolo.tsx always
  // forwards a single capture to the server regardless of what it detects,
  // rather than only opportunistically like it does during live detection).
  detect: (
    ctx: CanvasRenderingContext2D,
    opts: { isSingleCapture: boolean }
  ) => Promise<number>;
  currentModelResolution: number[];
  changeCurrentModelResolution: (width?: number, height?: number) => void;
  // When true, size the capture canvas to the webcam's native resolution
  // (video.videoWidth/videoHeight) instead of its on-page display size -
  // for modes that need the full-resolution frame (e.g. streaming full-res
  // JPEGs to a server) rather than whatever size the video happens to be
  // rendered at. Defaults to false (existing behavior) for every other
  // mode, which intentionally captures at display resolution since that's
  // all the model input (or the display overlay) needs.
  nativeResolutionCapture?: boolean;
}) => {
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [totalTime, setTotalTime] = useState<number>(0);
  const webcamRef = useRef<Webcam>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const liveDetection = useRef<boolean>(false);

  const [facingMode, setFacingMode] = useState<string>('environment');
  const originalSize = useRef<number[]>([0, 0]);

  const [modelResolution, setModelResolution] = useState<number[]>(
    props.currentModelResolution
  );

  useEffect(() => {
    setModelResolution(props.currentModelResolution);
  }, [props.currentModelResolution]);

  const capture = () => {
    const canvas = videoCanvasRef.current!;
    const context = canvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    if (facingMode === 'user') {
      context.setTransform(-1, 0, 0, 1, canvas.width, 0);
    }

    context.drawImage(
      webcamRef.current!.video!,
      0,
      0,
      canvas.width,
      canvas.height
    );

    if (facingMode === 'user') {
      context.setTransform(1, 0, 0, 1, 0, 0);
    }
    return context;
  };

  const runModel = async (
    ctx: CanvasRenderingContext2D,
    isSingleCapture: boolean
  ) => {
    if (!props.ready) return;
    const inferenceTime = await props.detect(ctx, { isSingleCapture });
    setInferenceTime(inferenceTime);
  };

  const runLiveDetection = async () => {
    if (liveDetection.current) {
      liveDetection.current = false;
      return;
    }
    if (!props.ready) return;
    liveDetection.current = true;
    while (liveDetection.current) {
      const startTime = Date.now();
      const ctx = capture();
      if (!ctx) return;
      await runModel(ctx, false);
      setTotalTime(Date.now() - startTime);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve())
      );
    }
  };

  const processImage = async () => {
    reset();
    const ctx = capture();
    if (!ctx) return;

    // create a copy of the canvas
    const boxCtx = document
      .createElement('canvas')
      .getContext('2d') as CanvasRenderingContext2D;
    boxCtx.canvas.width = ctx.canvas.width;
    boxCtx.canvas.height = ctx.canvas.height;
    boxCtx.drawImage(ctx.canvas, 0, 0);

    await runModel(boxCtx, true);
    ctx.drawImage(boxCtx.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  };

  const reset = async () => {
    var context = videoCanvasRef.current!.getContext('2d')!;
    context.clearRect(0, 0, originalSize.current[0], originalSize.current[1]);
    liveDetection.current = false;
  };

  const [SSR, setSSR] = useState<Boolean>(true);

  const setWebcamCanvasOverlaySize = () => {
    const element = webcamRef.current!.video!;
    if (!element) return;
    var w = props.nativeResolutionCapture ? element.videoWidth : element.offsetWidth;
    var h = props.nativeResolutionCapture ? element.videoHeight : element.offsetHeight;
    var cv = videoCanvasRef.current;
    if (!cv) return;
    cv.width = w;
    cv.height = h;
  };

  // close camera when browser tab is minimized
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        liveDetection.current = false;
      }
      // set SSR to true to prevent webcam from loading when tab is not active
      setSSR(document.hidden);
    };
    setSSR(document.hidden);
    document.addEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  if (SSR) {
    return <div>Loading...</div>;
  }

  return (
    <div className="flex flex-row flex-wrap w-full justify-evenly align-center">
      <div
        id="webcam-container"
        className="flex items-center justify-center webcam-container"
      >
        <Webcam
          mirrored={facingMode === 'user'}
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          imageSmoothing={true}
          videoConstraints={{
            facingMode: facingMode,
            // width: props.width,
            // height: props.height,
          }}
          onLoadedMetadata={() => {
            setWebcamCanvasOverlaySize();
            const video = webcamRef.current!.video!;
            originalSize.current = props.nativeResolutionCapture
              ? [video.videoWidth, video.videoHeight]
              : [video.offsetWidth, video.offsetHeight];
          }}
          forceScreenshotSourceSize={true}
        />
        <canvas
          id="cv1"
          ref={videoCanvasRef}
          style={{
            position: 'absolute',
            zIndex: 10,
            backgroundColor: 'rgba(0,0,0,0)',
            // Other modes size the canvas's width/height attributes to
            // match the video's already-shrunk on-page display size (via
            // Tailwind's video preflight), so the canvas's rendered size
            // equals its pixel buffer with no extra CSS needed. Native-
            // resolution mode's buffer is much larger than that (the raw
            // camera resolution), so it needs the same responsive-shrink
            // Tailwind gives <video> applied here explicitly.
            ...(props.nativeResolutionCapture
              ? { maxWidth: '100%', height: 'auto' }
              : {}),
          }}
        ></canvas>
      </div>
      <div className="flex flex-col items-center justify-center">
        {!props.ready && (
          <div className="flex flex-col items-center gap-2 m-3">
            {props.sessionError ? (
              <>
                <div className="text-red-500 text-center max-w-xs">
                  {props.sessionError}
                </div>
                <button
                  onClick={props.onRetrySession}
                  className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1"
                >
                  Retry
                </button>
              </>
            ) : props.retryingIn !== null ? (
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"
                  aria-hidden="true"
                ></span>
                <span>
                  Attempt {props.loadAttempt} of {props.maxLoadAttempts}{' '}
                  failed, retrying in {props.retryingIn}s...
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"
                  aria-hidden="true"
                ></span>
                <span>
                  Loading model (attempt {props.loadAttempt} of{' '}
                  {props.maxLoadAttempts})... ({props.secondsUntilTimeout}s)
                </span>
              </div>
            )}
          </div>
        )}
        <div className="flex flex-row flex-wrap items-center justify-center gap-1 m-5">
          <div className="flex items-stretch items-center justify-center gap-1">
            <button
              disabled={!props.ready}
              onClick={async () => {
                const startTime = Date.now();
                await processImage();
                setTotalTime(Date.now() - startTime);
              }}
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1 disabled:opacity-40 disabled:pointer-events-none"
            >
              Capture Photo
            </button>
            <button
              disabled={!props.ready}
              onClick={async () => {
                if (liveDetection.current) {
                  liveDetection.current = false;
                } else {
                  runLiveDetection();
                }
              }}
              //on hover, shift the button up
              className={`
              p-2  border-dashed border-2 rounded-xl hover:translate-y-1 disabled:opacity-40 disabled:pointer-events-none
              ${liveDetection.current ? 'bg-white text-black' : ''}

              `}
            >
              Live Detection
            </button>
          </div>
          <div className="flex items-stretch items-center justify-center gap-1">
            <button
              onClick={() => {
                reset();
                setFacingMode(facingMode === 'user' ? 'environment' : 'user');
              }}
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1 "
            >
              Switch Camera
            </button>
            <button
              onClick={() => {
                reset();
                props.changeCurrentModelResolution();
              }}
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1 "
            >
              Change Model
            </button>
            <button
              onClick={reset}
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1 "
            >
              Reset
            </button>
          </div>
        </div>
        {/* <div>
          <div>Yolov10 has a dynamic resolution with a maximum of 640x640</div>
          <div className="flex items-stretch items-center justify-center gap-1">
            <input
              value={modelResolution[0]}
              max={640}
              type="number"
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1"
              placeholder="Width"
              onChange={(e) => {
                setModelResolution([
                  parseInt(e.target.value),
                  modelResolution[1],
                ]);
              }}
            />
            <input
              value={modelResolution[1]}
              max={640}
              type="number"
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1"
              placeholder="Height"
              onChange={(e) => {
                setModelResolution([
                  modelResolution[0],
                  parseInt(e.target.value),
                ]);
              }}
            />
            <button
              onClick={() => {
                reset();
                if (modelResolution[0] > 640 || modelResolution[1] > 640) {
                  alert('Maximum resolution is 640x640');
                  return;
                }
                props.changeCurrentModelResolution(
                  modelResolution[0],
                  modelResolution[1]
                );
              }}
              className="p-2 border-2 border-dashed rounded-xl hover:translate-y-1"
            >
              Apply
            </button>
          </div>
        </div> */}
        <div>Using {props.modelName}</div>
        <div className="flex flex-row flex-wrap items-center justify-between w-full gap-3 px-5">
          <div>
            {'Model Inference Time: ' + inferenceTime.toFixed() + 'ms'}
            <br />
            {'Total Time: ' + totalTime.toFixed() + 'ms'}
            <br />
            {'Overhead Time: +' + (totalTime - inferenceTime).toFixed(2) + 'ms'}
          </div>
          <div>
            <div>
              {'Model FPS: ' + (1000 / inferenceTime).toFixed(2) + 'fps'}
            </div>
            <div>{'Total FPS: ' + (1000 / totalTime).toFixed(2) + 'fps'}</div>
            <div>
              {'Overhead FPS: ' +
                (1000 * (1 / totalTime - 1 / inferenceTime)).toFixed(2) +
                'fps'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;
