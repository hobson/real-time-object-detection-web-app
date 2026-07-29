import { renderHook } from '@testing-library/react';
import { useAutoAlprSubmit } from '../autoAlprSubmit';
import { Detection } from '../detectionTypes';
import { INFERENCE_ENDPOINT } from '../inferenceEndpoint';

const carDetection: Detection = { class: 'car', confidence: 0.9, box: [0, 0, 1, 1] };
const catDetection: Detection = { class: 'cat', confidence: 0.9, box: [0, 0, 1, 1] };

// canvas.toBlob is the one DOM API the hook touches directly - jsdom doesn't
// implement it, so stub it to synchronously hand back a fake Blob.
function makeMockCtx() {
  return {
    canvas: {
      toBlob: jest.fn((cb: (b: Blob | null) => void) => cb(new Blob(['x']))),
    },
  } as unknown as CanvasRenderingContext2D;
}

describe('useAutoAlprSubmit', () => {
  beforeEach(() => {
    // jsdom doesn't define fetch itself, so spyOn (which requires the
    // property to already exist) can't be used here - assign a fresh mock.
    global.fetch = jest.fn().mockResolvedValue({} as Response);
    // Start comfortably past MIN_SUBMIT_INTERVAL_MS (5s) so the very first
    // call in each test isn't itself rate-limited against lastSubmittedAt's
    // initial value of 0.
    jest.useFakeTimers().setSystemTime(10_000);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it('does not submit when no detection is a trigger class', () => {
    const { result } = renderHook(() => useAutoAlprSubmit());
    result.current(makeMockCtx(), [catDetection]);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('submits to /alpr/predict when a trigger-class detection is present', () => {
    const { result } = renderHook(() => useAutoAlprSubmit());
    result.current(makeMockCtx(), [carDetection]);
    expect(global.fetch).toHaveBeenCalledWith(
      `${INFERENCE_ENDPOINT}/alpr/predict`,
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('rate-limits repeated submissions within MIN_SUBMIT_INTERVAL_MS', () => {
    const { result } = renderHook(() => useAutoAlprSubmit());
    result.current(makeMockCtx(), [carDetection]);
    jest.setSystemTime(11_000); // 1s later - still within the 5s floor
    result.current(makeMockCtx(), [carDetection]);
    expect(global.fetch).toHaveBeenCalledTimes(1);

    jest.setSystemTime(16_000); // past the 5s floor
    result.current(makeMockCtx(), [carDetection]);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it('force=true bypasses both the trigger-class check and the rate limit', () => {
    const { result } = renderHook(() => useAutoAlprSubmit());
    result.current(makeMockCtx(), [catDetection], true);
    result.current(makeMockCtx(), [catDetection], true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});
