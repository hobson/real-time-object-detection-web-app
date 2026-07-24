import { env, InferenceSession, Tensor } from 'onnxruntime-web';

// onnxruntime-web resolves .wasm files relative to its own script URL by
// default, which mobile browsers (e.g. Firefox/Fennec) resolve differently
// than desktop and can fail entirely ("no available backend found"). Point
// it explicitly at where next.config.js's CopyPlugin actually puts them.
env.wasm.wasmPaths = '/runtime/';
// The threaded wasm backend needs SharedArrayBuffer, which requires
// cross-origin-isolation headers (COOP/COEP) this app doesn't send. Without
// them it's unreliable across browsers (especially mobile Firefox), so
// force single-threaded execution instead of depending on isolation headers.
env.wasm.numThreads = 1;

export async function createModelCpu(url: string): Promise<InferenceSession> {
  return await InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

export async function runModel(
  model: InferenceSession,
  preprocessedData: Tensor
): Promise<[Tensor, number]> {
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = preprocessedData;
    const start = Date.now();
    const outputData = await model.run(feeds);
    const end = Date.now();
    const inferenceTime = end - start;
    const output = outputData[model.outputNames[0]];
    return [output, inferenceTime];
  } catch (e) {
    console.error(e);
    throw e;
  }
}
