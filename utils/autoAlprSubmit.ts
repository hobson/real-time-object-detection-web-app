import { useRef } from 'react';
import { INFERENCE_ENDPOINT } from './inferenceEndpoint';
import { Detection } from './detectionTypes';

// Client-side detection (Yolo.tsx) used to only draw boxes locally - it
// never told the backend what it saw. This hook opportunistically forwards
// a frame (as multipart, with the client's own YOLO detections attached as
// `client_detections` metadata - see request_parsing.py/persist.py)
// whenever a road-relevant object shows up, so the server logs the
// submission (submitted_images/detection_labels, see persist.py) with both
// what the browser found AND runs its own plate detection + OCR pass on
// it. There's no COCO "license plate" class, so these trigger classes are
// used as a proxy - the ALPR endpoint runs its own plate detector over the
// whole image, it doesn't need the browser to have found a plate first.
const TRIGGER_CLASSES = new Set([
  'car',
  'truck',
  'bus',
  'motorbike',
  'bicycle',
  'person',
]);

// This only needs to happen "sometimes", not on every frame: /alpr/predict
// does its own (expensive, OCR-included) detection pass server-side, while
// the browser's live-detection loop runs at tens of fps - without a floor
// here it would fire on nearly every frame a trigger-class object stays in
// view.
const MIN_SUBMIT_INTERVAL_MS = 5_000;

export function useAutoAlprSubmit() {
  const lastSubmittedAt = useRef<number>(0);

  // `force`: bypass both the trigger-class check and the rate limit - set
  // by the explicit "Capture Photo" button (see ObjectDetectionCamera's
  // isSingleCapture), which should always reach the server since the user
  // deliberately asked for that one frame, unlike the opportunistic
  // best-effort sampling live detection does on every frame.
  return (ctx: CanvasRenderingContext2D, detections: Detection[], force = false) => {
    if (!force && !detections.some((d) => TRIGGER_CLASSES.has(d.class))) return;

    const now = Date.now();
    if (!force && now - lastSubmittedAt.current < MIN_SUBMIT_INTERVAL_MS) return;
    lastSubmittedAt.current = now;

    ctx.canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const form = new FormData();
        form.append('image', blob, 'frame.jpg');
        form.append('metadata', JSON.stringify({ client_detections: detections }));
        fetch(`${INFERENCE_ENDPOINT}/alpr/predict`, {
          method: 'POST',
          body: form,
        }).catch((e) =>
          console.error('Failed to auto-submit frame for ALPR', e)
        );
      },
      'image/jpeg',
      0.85
    );
  };
}
