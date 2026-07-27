// Shape of one YOLO detection - what a model's postprocess step produces,
// and what a client posts to the backend as `client_detections` (box
// normalized 0-1 against canvas size either way - see persist.py's
// _detection_label, which accepts this same class/confidence/box shape).
export type Detection = { class: string; confidence: number; box: number[] };
