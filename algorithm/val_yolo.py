import time
import numpy as np
from ultralytics import YOLO

DATA    = "C:/Projects_H.W/FINAL-PROJECT/UniView/AI/datasets/MATRIX_yolo_format/MATRIX.yaml"
FP32_OV = "C:/Projects_H.W/FINAL-PROJECT/UniView/algorithm/weights/best_openvino_model"
INT8_OV = "C:/Projects_H.W/FINAL-PROJECT/UniView/algorithm/weights/best_int8_openvino_model"

DUMMY = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 4  # batch of 4 frames
RUNS  = 20


def bench(name: str, weights: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    model = YOLO(weights, task="detect")

    # warmup
    for _ in range(3):
        model.predict(DUMMY, conf=0.5, iou=0.45, classes=[0],
                      verbose=False, imgsz=640, device="cpu")

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict(DUMMY, conf=0.5, iou=0.45, classes=[0],
                      verbose=False, imgsz=640, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    print(f"  Batch=4  avg={avg:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms  ({RUNS} runs)")

    print("  Validating mAP...")
    metrics = model.val(data=DATA, imgsz=640, device="cpu", verbose=False)
    print(f"  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


bench("OpenVINO FP32", FP32_OV)
bench("OpenVINO INT8", INT8_OV)

print("\nDone.")

