# Subscribing to per-object-type detection notifications

The app can push a phone notification — with the detected frame attached
as a photo — whenever it spots specific kinds of objects. Rather than one
firehose of "something happened" notifications, each object type has its
**own ntfy topic**, so you can subscribe to just the ones you care about
(e.g. only `car`, or only `people`).

This is separate from actually running detection — see
[`realtime-object-detection.md`](./realtime-object-detection.md) for how
to use the camera/detection UI itself. Notifications fire automatically
whenever the app detects a matching object, in either detection mode
(In-browser or Server-side), rate-limited to once per topic per 30
seconds.

## Available topics

| Object type | Topic | Notes |
|---|---|---|
| person | `object-detection-people` | |
| car | `object-detection-car` | |
| laptop | `object-detection-laptop` | |
| flower | `object-detection-flower` | Mapped from the model's `pottedplant` class — the closest match available. No model currently distinguishes flowers from potted plants generally. |
| dog | `object-detection-dog` | |
| cat | `object-detection-cat` | |
| bicycle | `object-detection-bicycle` | |
| truck | `object-detection-truck` | |
| bus | `object-detection-bus` | |
| bird | `object-detection-bird` | |
| license plate | `object-detection-license_plate` | **Not live yet.** No current model detects license plates — see [`PLAN-realtime-license-plate-detection.md`](./PLAN-realtime-license-plate-detection.md) for the dataset and fine-tuning plan to add this. The topic name is reserved for when that model ships; don't subscribe expecting notifications yet. |
| tree | — | **Not supported.** Trees aren't one of the model's 80 object classes and there's no reasonably close substitute (unlike flower→pottedplant), so no topic exists for this. |

Want a different object type added? The model already recognizes 80
classes total (see [`data/yolo_classes.ts`](../data/yolo_classes.ts) for
the full list) — most of them could get their own topic by adding one
entry to `CLASS_TO_TOPIC` in
[`utils/notify.ts`](../utils/notify.ts).

## How to subscribe

1. Install the **ntfy** app
   ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) /
   [iOS](https://apps.apple.com/us/app/ntfy/id1625396347)), or use
   [the web UI](https://ntfy.sh/app) in a browser.
2. Add a subscription for **each** object type you want notifications for:
   - **Server:** `https://taco.tail9f615d.ts.net`
   - **Username:** `subscriber`
   - **Password:** the read-only ntfy token (see this repo's local `.env`
     — gitignored, ask whoever administers the server if you don't have
     it)
   - **Topic:** one of the topic names from the table above, e.g.
     `object-detection-people`

You can subscribe to as many or as few topics as you like — each is
independent. To get notified for *every* supported object type without
adding each one by hand, some ntfy clients support wildcard topic
subscriptions (`object-detection-*`); check your client's docs, since
support varies.

## Notes

- The same `subscriber` read-only token works for all `object-detection-*`
  topics — it was granted wildcard access, not per-topic, so no new setup
  is needed as new object types are added later.
- If multiple object types are detected in the same frame (e.g. both a
  person and a car), you'll get a separate notification per topic you're
  subscribed to, not one combined notification — each topic's rate limit
  is tracked independently, so a busy `people` topic won't suppress a
  `car` notification arriving around the same time.
