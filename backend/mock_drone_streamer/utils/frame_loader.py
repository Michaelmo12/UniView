from pathlib import Path
import cv2


def load_frame(drone_id, frame_num, dataset_path):
    # Construct path: image_subsets/D{1-8}/{0000-0999}.png
    folder_name = f"D{drone_id}"
    frame_path = Path(dataset_path) / "image_subsets" / folder_name / f"{frame_num:04d}.png"

    if not frame_path.exists():
        return None

    # Load image
    frame = cv2.imread(str(frame_path))
    return frame
