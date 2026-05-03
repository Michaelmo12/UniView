from ultralytics.utils.benchmarks import benchmark

SMALL_PT = "C:/Projects_H.W/FINAL-PROJECT/UniView/algorithm/weights/best.pt"

benchmark(model=SMALL_PT, imgsz=640, half=False, device="cpu")

