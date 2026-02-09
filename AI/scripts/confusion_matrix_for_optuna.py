from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("E:/projectants/UniView/AI/models/trained/optuna/trial_1/weights/best.pt")
    model.val(data="E:/projectants/UniView/AI/datasets/MATRIX_yolo_format/MATRIX.yaml", plots=True)