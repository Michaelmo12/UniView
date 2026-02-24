"""
inference.py - Run detection on an image using the trained model
"""

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load model
model = YOLO("E:/projectants/UniView/AI/models/trained/final/weights/best.pt")

# Run on image - D4 frame 100
results = model.predict(
    "E:/projectants/UniView/AI/datasets/MATRIX_yolo_format/D4/0200.png",
    conf=0.25,
)

# Get original image
img = results[0].orig_img.copy()

# Create annotator with custom font size
annotator = Annotator(img, line_width=1, font_size=20)

# Draw boxes manually
for box in results[0].boxes:
    xyxy = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    label = f"person {conf:.2f}"
    annotator.box_label(xyxy, label, color=(255, 0, 0))

img = annotator.result()

# Resize to fit screen (50% of original)
scale = 0.75
img = cv2.resize(img, None, fx=scale, fy=scale)

# Show
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
