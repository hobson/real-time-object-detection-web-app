"""Export a .pt checkpoint to ONNX for tinygrad (openpilot on-device inference),
as opposed to main.py's export target (onnxruntime-web in the browser).

The difference is `nms=False`: tinygrad's ONNX importer (openpilot's
tinygrad_repo/tinygrad/nn/onnx.py) has no NonMaxSuppression op, so the
nms=True export main.py produces (see ultralytics_pt_to_onnx.md) loads and
runs right up to the final NMS node and then throws NotImplementedError.
Exporting without nms=True gives the raw per-anchor box+class-score tensor
instead - every op in that graph (Conv/MatMul/Sigmoid/Softmax/Reshape/Split/
Concat/Resize/...) is supported by tinygrad's importer. NMS then happens as
a plain numpy postprocessing step - see postprocess_tinygrad.py.

Usage:
  python export_for_tinygrad.py <path/to/model.pt> [--imgsz 256] [--opset 19]
"""
import argparse

from ultralytics import YOLO


def export_for_tinygrad(weights_path: str, imgsz: int, opset: int) -> str:
  model = YOLO(weights_path)
  return model.export(format="onnx", simplify=True, dynamic=False, imgsz=imgsz, batch=1, nms=False, opset=opset)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("weights", help="path to the .pt checkpoint, e.g. runs/detect/.../best.pt")
  parser.add_argument("--imgsz", type=int, default=256)
  parser.add_argument("--opset", type=int, default=19)
  args = parser.parse_args()

  out_path = export_for_tinygrad(args.weights, args.imgsz, args.opset)
  print(f"exported to {out_path}")
