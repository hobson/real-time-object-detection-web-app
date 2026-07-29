import { postprocessYolov10, postprocessYolov7 } from '../Yolo';
import { Detection } from '../../../utils/detectionTypes';

const classNames = ['cat', 'dog', 'car'];

// postprocessYolov10/Yolov7 only touch a handful of CanvasRenderingContext2D
// members - a plain object satisfies them without pulling in jsdom's canvas
// (which isn't implemented) or a canvas-mocking library.
function makeMockCtx(width = 100, height = 100) {
  return {
    canvas: { width, height },
    clearRect: jest.fn(),
    strokeRect: jest.fn(),
    fillRect: jest.fn(),
    fillText: jest.fn(),
    strokeStyle: '',
    fillStyle: '',
    lineWidth: 0,
    font: '',
  } as unknown as CanvasRenderingContext2D;
}

// Same shape onnxruntime-web's Tensor exposes (dims + a flat typed array) -
// postprocess only reads those two fields, so a plain object stands in.
function makeTensor(dims: number[], data: number[]) {
  return { dims, data: new Float32Array(data) } as any;
}

// Every postprocess function reports its result via an onDetections
// callback rather than a return value - this runs one and captures what it
// passed, so call sites can just do `run(postprocessX, ...)`.
function run(
  postprocess: typeof postprocessYolov10 | typeof postprocessYolov7,
  ctx: CanvasRenderingContext2D,
  modelResolution: number[],
  tensor: any,
  classes: string[]
): Detection[] {
  let detections: Detection[] = [];
  postprocess(ctx, modelResolution, tensor, () => 'red', (d) => (detections = d), classes);
  return detections;
}

describe('postprocessYolov10', () => {
  it('emits a detection per row above MIN_CONFIDENCE and stops at the first below it (pre-sorted input)', () => {
    // [x0, y0, x1, y1, score, cls_id] per row, flattened into [1, N, 6].
    const rows = [
      [10, 10, 50, 50, 0.9, 2], // car, kept
      [0, 0, 20, 20, 0.6, 0], // cat, kept
      [0, 0, 5, 5, 0.3, 1], // below MIN_CONFIDENCE (0.5) - loop breaks here
      [0, 0, 5, 5, 0.99, 1], // would pass on its own, but never reached
    ].flat();
    const tensor = makeTensor([1, rows.length, 6], rows);

    const detections = run(postprocessYolov10, makeMockCtx(), [100, 100], tensor, classNames);

    expect(detections).toHaveLength(2);
    expect(detections[0].class).toBe('car');
    expect(detections[0].confidence).toBeCloseTo(0.9);
    expect(detections[1].class).toBe('cat');
  });

  it('normalizes box coordinates to 0-1 against canvas size, after scaling from model resolution', () => {
    // Model resolution is 50x50, canvas is 100x100 -> dx=dy=2.
    const tensor = makeTensor([1, 6, 6], [10, 10, 30, 30, 0.8, 2]);

    const detections = run(postprocessYolov10, makeMockCtx(100, 100), [50, 50], tensor, classNames);

    // x0=10*2=20 -> 20/100=0.2, x1=30*2=60 -> 0.6 (same for y).
    expect(detections[0].box).toEqual([0.2, 0.2, 0.6, 0.6]);
  });

  it('reports no detections when every row is below MIN_CONFIDENCE', () => {
    const tensor = makeTensor([1, 6, 6], [0, 0, 5, 5, 0.1, 0]);

    const detections = run(postprocessYolov10, makeMockCtx(), [100, 100], tensor, classNames);

    expect(detections).toEqual([]);
  });
});

describe('postprocessYolov7', () => {
  it('emits a detection per row above MIN_CONFIDENCE, skipping (not breaking on) low-score rows since input is unsorted', () => {
    // [batch_id, x0, y0, x1, y1, cls_id, score] per row, flattened into [N, 7].
    const rows = [
      [0, 0, 0, 5, 5, 1, 0.2], // below threshold - skipped, not a break
      [0, 10, 10, 50, 50, 2, 0.9], // car, kept
      [0, 0, 0, 20, 20, 0, 0.6], // cat, kept
    ].flat();
    const tensor = makeTensor([3, 7], rows);

    const detections = run(postprocessYolov7, makeMockCtx(), [100, 100], tensor, classNames);

    expect(detections).toHaveLength(2);
    expect(detections.map((d) => d.class)).toEqual(['car', 'cat']);
  });

  it('uses the single-class array for the license-plate model (cls_id always 0)', () => {
    const plateClasses = ['license_plate'];
    const tensor = makeTensor([1, 7], [0, 10, 10, 50, 50, 0, 0.75]);

    const detections = run(postprocessYolov7, makeMockCtx(), [384, 384], tensor, plateClasses);

    expect(detections[0].class).toBe('license_plate');
  });
});
