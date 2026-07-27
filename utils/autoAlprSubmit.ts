import { useRef } from 'react';
import { INFERENCE_ENDPOINT } from './inferenceEndpoint';

// Client-side detection (Yolo.tsx) only draws boxes locally - it never tells
// the backend what it saw. This hook opportunistically forwards a frame to
// POST /alpr/predict whenever a car or person shows up, so the server logs
// the submission (submitted_images/detection_labels, see persist.py) and
// runs its own plate detection + OCR pass on it. There's no COCO "license
// plate" class, so "car" or "person" in frame is used as a proxy trigger -
// the ALPR endpoint runs its own plate detector over the whole image, it
// doesn't need the browser to have found a plate first.
const TRIGGER_CLASSES = new Set(['car', 'person']);

// This only needs to happen "sometimes", not on every frame: /alpr/predict
// does its own (expensive, OCR-included) detection pass server-side, while
// the browser's live-detection loop runs at tens of fps - without a floor
// here it would fire on nearly every frame a car/person stays in view.
const MIN_SUBMIT_INTERVAL_MS = 5_000;

export function useAutoAlprSubmit() {
  const lastSubmittedAt = useRef<number>(0);

  // `force`: bypass both the trigger-class check and the rate limit - set
  // by the explicit "Capture Photo" button (see ObjectDetectionCamera's
  // isSingleCapture), which should always reach the server since the user
  // deliberately asked for that one frame, unlike the opportunistic
  // best-effort sampling live detection does on every frame.
  return (
    ctx: CanvasRenderingContext2D,
    detectedClasses: string[],
    force = false
  ) => {
    if (!force && !detectedClasses.some((c) => TRIGGER_CLASSES.has(c))) return;

    const now = Date.now();
    if (!force && now - lastSubmittedAt.current < MIN_SUBMIT_INTERVAL_MS) return;
    lastSubmittedAt.current = now;

    ctx.canvas.toBlob(
      (blob) => {
        if (!blob) return;
        fetch(`${INFERENCE_ENDPOINT}/alpr/predict`, {
          method: 'POST',
          body: blob,
          headers: { 'Content-Type': 'image/jpeg' },
        }).catch((e) =>
          console.error('Failed to auto-submit frame for ALPR', e)
        );
      },
      'image/jpeg',
      0.85
    );
  };
}
