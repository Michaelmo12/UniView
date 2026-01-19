# Models Directory

This folder contains the YOLO models used for object detection in the UniView project.

## Structure

```
models/
├── pretrained/    # Pre-trained YOLO models from Ultralytics
│   ├── yolo11s.pt # YOLOv11 Small model
│   └── yolo11m.pt # YOLOv11 Medium model
└── trained/       # Fine-tuned models on custom datasets
    └── (Your trained models will be saved here)
```

## Pretrained Models

The `pretrained/` directory contains base YOLO models:
- **yolo11s.pt** - YOLOv11 Small: Faster inference, lower accuracy
- **yolo11m.pt** - YOLOv11 Medium: Balanced speed and accuracy

These models are used as starting points for training on custom datasets and fine tuning.

## Trained Models

The `trained/` directory will contain:
- Models fine-tuned on the VisDrone dataset
- Custom trained models for specific detection tasks
- Best checkpoints from training sessions

## Usage

Models are loaded and used in [yolomodel.py](../yolomodel.py) for training and inference.