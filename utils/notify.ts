import { useRef } from 'react';

// Classes that trigger a push notification, and which per-object-type ntfy
// topic they map to (as the suffix after "object-detection-" - the prefix
// is applied server-side by notify-proxy on taco, see NTFY_TOPIC_PREFIX
// there). Multiple classes can share a topic (e.g. no COCO "flower" class,
// so "pottedplant" is mapped there as the closest visual analog).
//
// "tree" and "license_plate" were requested but have no COCO class and no
// model that currently detects them - left out rather than mapped to
// something misleading. license_plate is planned; see
// docs/PLAN-realtime-license-plate-detection.md for the model work needed
// first.
export const CLASS_TO_TOPIC: Record<string, string> = {
  person: 'people',
  car: 'car',
  laptop: 'laptop',
  pottedplant: 'flower',
  dog: 'dog',
  cat: 'cat',
  bicycle: 'bicycle',
  truck: 'truck',
  bus: 'bus',
  bird: 'bird',
};

// Public CORS-enabled proxy on taco (see /notify-proxy) that forwards to
// ntfy - called directly from the browser so the ntfy token never has to
// live in this app's client bundle or server.
export const NOTIFY_ENDPOINT =
  process.env.NEXT_PUBLIC_NOTIFY_URL ||
  'https://taco.tail9f615d.ts.net:8443/notify';

const NOTIFY_MIN_INTERVAL_MS = 30_000;

// Rate-limited per topic (not globally) so a busy topic (e.g. "people")
// doesn't suppress notifications for an unrelated one (e.g. "car") that
// happens to be in the same frame.
export function useNotifyDetection() {
  const lastNotifiedAt = useRef<Map<string, number>>(new Map());

  return (ctx: CanvasRenderingContext2D, detectedClasses: string[]) => {
    const topicToClasses = new Map<string, Set<string>>();
    for (const c of detectedClasses) {
      const topic = CLASS_TO_TOPIC[c];
      if (!topic) continue;
      if (!topicToClasses.has(topic)) topicToClasses.set(topic, new Set());
      topicToClasses.get(topic)!.add(c);
    }
    if (topicToClasses.size === 0) return;

    const now = Date.now();
    const topicsToNotify = Array.from(topicToClasses.entries()).filter(
      ([topic]) =>
        now - (lastNotifiedAt.current.get(topic) ?? 0) >= NOTIFY_MIN_INTERVAL_MS
    );
    if (topicsToNotify.length === 0) return;
    topicsToNotify.forEach(([topic]) => lastNotifiedAt.current.set(topic, now));

    ctx.canvas.toBlob(
      (blob) => {
        if (!blob) return;
        for (const [topic, classes] of topicsToNotify) {
          fetch(
            `${NOTIFY_ENDPOINT}?title=${encodeURIComponent(
              Array.from(classes).join(', ') + ' detected'
            )}&tags=camera&topic=${encodeURIComponent(topic)}`,
            { method: 'POST', body: blob }
          ).catch((e) =>
            console.error(`Failed to send ntfy notification (${topic})`, e)
          );
        }
      },
      'image/jpeg',
      0.8
    );
  };
}
