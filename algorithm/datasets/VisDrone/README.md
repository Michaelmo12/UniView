# VisDrone Dataset

This folder contains the VisDrone dataset in YOLO format for object detection training.

## Structure

```
visdrone/
├── data.yaml      # Dataset configuration (paths, classes)
├── images/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images
└── labels/
    ├── train/     # Training labels (YOLO format)
    ├── val/       # Validation labels (YOLO format)
    └── test/      # Test labels (YOLO format)
```

## Configuration

The `data.yaml` file contains:
- Dataset paths (train, val, test)
- Number of classes
- Class names

## Format

Labels are in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).
