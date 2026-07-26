// Base URL of the FastAPI inference-server (see CLAUDE.md's "Server-side
// inference"), shared by every component that talks to it (YoloServer.tsx,
// AlprServer.tsx, and the local Yolo.tsx auto-submit hook) so the override
// env var and default only need to be declared once.
export const INFERENCE_ENDPOINT =
  process.env.NEXT_PUBLIC_INFERENCE_URL ||
  'https://taco.tail9f615d.ts.net:8443/infer';
