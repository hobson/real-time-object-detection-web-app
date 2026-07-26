# Convert YOLO models from ultralytics .pt to .onnx format that is compatible for onnxruntime webassembly

To retrieve a YOLO model from ultralytics, you need to install the `ultralytics` package. You can do this by running:

```bash
pip install ultralytics
```

To retrieve a YOLO model from ultralytics, you can use the following code:

```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov10n.pt")

# Export the model to ONNX format
model.export(format="onnx", simplify=True, dynamic=True)

```

This will save the model in the `yolov10n.onnx` file. However, this model is not compatible for onnxruntime webassembly.

## Preferred export for models whose native output isn't already NMS'd

`yolo11n`/`yolo12n`'s raw ultralytics ONNX export is `[1, 84, anchors]` (box
coords + 80 raw class scores per anchor, not yet NMS'd) - handling that in
the browser meant per-anchor score scanning plus a hand-rolled NMS/IoU
implementation in `Yolo.tsx`, run every frame. `yolov10n` and
`yolov7-tiny`'s ultralytics exports already bake NMS into the graph and
output pre-filtered/sorted `[x0,y0,x1,y1,score,cls_id]` rows instead - much
less client-side work.

Ultralytics can do the same for yolo11/yolo12 via `nms=True`. Also prefer a
static shape (`dynamic=False`) here: `Yolo.tsx` always feeds a fixed-size
tensor (whatever `modelResolution` currently is) anyway, so dynamic
axes are pure overhead with no actual flexibility used. Pin `opset`
explicitly rather than trusting whatever your installed ultralytics
defaults to, so a later `pip install -U ultralytics` doesn't silently ship
a newer opset than the currently-tested `onnxruntime-web` version supports
(check `package.json`'s pinned version before bumping):

```python
from ultralytics import YOLO

model = YOLO("yolo12n.pt")
model.export(format="onnx", simplify=True, dynamic=False, imgsz=256, batch=1, nms=True, opset=19)
```

Output is `[1, 300, 6]` (300 zero-padded, descending-confidence rows of
`[x0, y0, x1, y1, score, cls_id]` in pixel coordinates) - exactly the shape
`postprocessYolov10` in `Yolo.tsx` already parses, so no new postprocess
function is needed; just point the model name at `postprocessYolov10` in
`postprocessMap`. Verify numerically before shipping - run the same input
through the `.pt` (raw ultralytics predict) and the exported `.onnx`
(`onnxruntime.InferenceSession`) and compare detections, and separately
load the `.onnx` through the actual `onnxruntime-web` package (not just
Python `onnxruntime`) to catch any WASM-specific incompatibility, e.g.:

```js
const ort = require('onnxruntime-web');
ort.env.wasm.numThreads = 1; // Node has no multi-threaded wasm; browsers do
const session = await ort.InferenceSession.create('models/yolo12n.onnx');
```

To convert the model to the compatible ONNX format, you can use the `onnxruntime` package. First, install the package:

```bash
pip install onnxruntime
```

Then run in the terminal:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort yolov10n.onnx --save_optimized_onnx_model
```

This will save the optimized model in the `yolov10n.ort` file as well as the optimized ONNX model in the `yolov10n.optimized.onnx` file. Either of these files can be used in the web app.

# References

- [Onnxuntime Conversion Guide](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
- [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export)
