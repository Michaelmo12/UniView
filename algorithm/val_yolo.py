from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="C:\\Projects_H.W\\FINAL-PROJECT\\UniView\\algorithm\\weights\\best.pt", data="coco8.yaml", imgsz=640, half=False, device='cpu', format="openvino")

# Benchmark specific export format
#benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, format="onnx")