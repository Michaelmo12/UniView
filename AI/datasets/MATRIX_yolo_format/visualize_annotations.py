"""
MATRIX Dataset Annotation Visualizer

Visualizes bounding box annotations on images to verify correctness.
Displays person IDs and bounding boxes for a specific drone/camera view.
"""

import json
import cv2
import matplotlib.pyplot as plt

from pathlib import Path


class MATRIXAnnotationVisualizer:
    """Visualizes MATRIX dataset annotations on images."""

    def __init__(self, dataset_root):
        """
        Initialize visualizer.

        Args:
            dataset_root: Path to MATRIX_yolo_format directory (working directory)
        """
        self.dataset_root = Path(dataset_root)
        # Annotations are in MATRIX_30x30/MATRIX_30x30/annotations_positions/
        self.annotations_dir = (
            self.dataset_root.parent.parent.parent
            / "MATRIX_30x30"
            / "MATRIX_30x30"
            / "annotations_positions"
        )
        # Images are in MATRIX_yolo_format/D{drone_id}/
        self.images_dir = self.dataset_root

    def load_image(self, drone_id, frame_num):
        """
        Load image for a specific drone and frame.

        Args:
            drone_id: Drone ID (1-8)
            frame_num: Frame number (0-999)

        Returns:
            numpy array: Image in BGR format (OpenCV), or None if not found
        """
        img_path = self.images_dir / f"D{drone_id}" / f"{frame_num:04d}.png"

        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Failed to load image from {img_path}")
            return None

        return img

    def load_annotations(self, frame_num):
        """
        Load annotations for a specific frame.

        Args:
            frame_num: Frame number (0-999)

        Returns:
            list: List of person annotations, or None if not found
        """
        json_path = self.annotations_dir / f"{frame_num:04d}.json"

        if not json_path.exists():
            print(f"Error: Annotations not found at {json_path}")
            return None

        try:
            with open(json_path, "r") as f:
                annotations = json.load(f)
            return annotations
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from {json_path}: {e}")
            return None

    def load_bboxes_for_frame(self, drone_id, frame_num):
        """
        Load bboxes for a specific drone and frame.

        Reads from filtered YOLO labels first (if they exist),
        otherwise falls back to original JSON annotations.

        Args:
            drone_id: Drone ID (1-8)
            frame_num: Frame number (0-999)

        Returns:
            list: List of (person_id, bbox) tuples for visible persons
                  bbox format: (xmin, ymin, xmax, ymax)
        """
        # Try to load from YOLO labels first (filtered)
        labels_dir = self.images_dir / f"D{drone_id}" / "labels"
        label_path = labels_dir / f"{frame_num:04d}.txt"

        if label_path.exists():
            # Read from filtered YOLO labels
            return self._load_from_yolo_labels(drone_id, label_path)
        else:
            # Fall back to original JSON annotations
            annotations = self.load_annotations(frame_num)
            if annotations is None:
                return []
            return self._filter_annotations_for_drone(annotations, drone_id)

    def _load_from_yolo_labels(self, drone_id, label_path):
        """Load bboxes from YOLO label file."""
        # Get image dimensions
        img = self.load_image(drone_id, int(label_path.stem))
        if img is None:
            return []

        img_height, img_width = img.shape[:2]
        visible_persons = []

        with open(label_path, "r") as f:
            for idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class_id, x_center, y_center, width, height (normalized)
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height

                    xmin = int(x_center - w / 2)
                    ymin = int(y_center - h / 2)
                    xmax = int(x_center + w / 2)
                    ymax = int(y_center + h / 2)

                    visible_persons.append((idx, (xmin, ymin, xmax, ymax)))

        return visible_persons

    def _filter_annotations_for_drone(self, annotations, drone_id):
        """
        Filter annotations for a specific drone/camera from JSON.

        Args:
            annotations: List of person annotations
            drone_id: Drone ID (1-8)

        Returns:
            list: List of (person_id, bbox) tuples for visible persons
                  bbox format: (xmin, ymin, xmax, ymax)
        """
        view_num = drone_id - 1  # Drone 1 = viewNum 0, Drone 2 = viewNum 1, etc.
        visible_persons = []

        for person in annotations:
            person_id = person.get("personID")

            # Find the view for this drone
            for view in person.get("views", []):
                if view.get("viewNum") == view_num:
                    xmin = view.get("xmin")
                    ymin = view.get("ymin")
                    xmax = view.get("xmax")
                    ymax = view.get("ymax")

                    # Skip if not visible (xmin == -1)
                    if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                        continue

                    # Validate bbox
                    if xmin >= xmax or ymin >= ymax:
                        print(
                            f"Warning: Invalid bbox for person {person_id} (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})"
                        )
                        continue

                    visible_persons.append((person_id, (xmin, ymin, xmax, ymax)))
                    break

        return visible_persons

    def draw_bboxes(self, img, visible_persons):
        """
        Draw bounding boxes and person IDs on image.

        Args:
            img: Image array (BGR format)
            visible_persons: List of (person_id, bbox) tuples

        Returns:
            numpy array: Image with bboxes drawn
        """
        img_annotated = img.copy()

        for person_id, (xmin, ymin, xmax, ymax) in visible_persons:
            # Draw bounding box (green, thickness 2)
            cv2.rectangle(img_annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Prepare label text
            label = f"Person {person_id}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text (filled green)
            cv2.rectangle(
                img_annotated,
                (xmin, ymin - text_height - baseline - 5),
                (xmin + text_width, ymin),
                (0, 255, 0),
                -1,  # Filled
            )

            # Draw text (black on green background)
            cv2.putText(
                img_annotated,
                label,
                (xmin, ymin - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2,
            )

        return img_annotated

    def visualize(self, drone_id, frame_num, save_path=None, show=True):
        """
        Visualize annotations for a specific drone and frame.

        Args:
            drone_id: Drone ID (1-8)
            frame_num: Frame number (0-999)
            save_path: Optional path to save the visualization
            show: Whether to display the image (default: True)

        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 70)
        print("Visualizing MATRIX Annotations")
        print("=" * 70)
        print(f"Drone: {drone_id}")
        print(f"Frame: {frame_num:04d}")
        print()

        # Load image
        img = self.load_image(drone_id, frame_num)
        if img is None:
            return False

        print(f"✓ Loaded image: {img.shape[1]}x{img.shape[0]} pixels")

        # Load bboxes (from YOLO labels or JSON)
        visible_persons = self.load_bboxes_for_frame(drone_id, frame_num)

        # Also load annotations to get total count
        annotations = self.load_annotations(frame_num)
        total_persons = len(annotations) if annotations else 0

        print(f"✓ Loaded annotations: {total_persons} persons total")
        print(f"✓ Found {len(visible_persons)} persons visible in Drone {drone_id}")
        print()

        if len(visible_persons) == 0:
            print("No persons visible in this camera view.")
            if show:
                # Still show the image even if no annotations
                self._display_image(img, drone_id, frame_num, 0)
            return True

        # Draw bboxes
        img_annotated = self.draw_bboxes(img, visible_persons)

        # Print person details
        print("Detected persons:")
        for person_id, (xmin, ymin, xmax, ymax) in visible_persons:
            width = xmax - xmin
            height = ymax - ymin
            print(
                f"  Person {person_id:3d}: bbox=({xmin:4d}, {ymin:4d}, {xmax:4d}, {ymax:4d}), size={width}x{height}px"
            )
        print()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_annotated)
            print(f"✓ Saved visualization to: {save_path}")
            print()

        # Display if requested
        if show:
            self._display_image(
                img_annotated, drone_id, frame_num, len(visible_persons)
            )

        return True

    def _display_image(self, img, drone_id, frame_num, num_persons):
        """
        Display image using matplotlib.

        Args:
            img: Image in BGR format
            drone_id: Drone ID for title
            frame_num: Frame number for title
            num_persons: Number of persons detected
        """
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        plt.figure(figsize=(16, 9))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(
            f"Drone {drone_id} | Frame {frame_num:04d} | Persons: {num_persons}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def visualize_multiple(self, drone_id, frame_nums, save_dir=None):
        """
        Visualize multiple frames for a drone.

        Args:
            drone_id: Drone ID (1-8)
            frame_nums: List of frame numbers to visualize
            save_dir: Optional directory to save visualizations
        """
        print("=" * 70)
        print(f"Visualizing Multiple Frames - Drone {drone_id}")
        print("=" * 70)
        print()

        for frame_num in frame_nums:
            if save_dir:
                save_path = Path(save_dir) / f"drone{drone_id}_frame{frame_num:04d}.png"
            else:
                save_path = None

            success = self.visualize(
                drone_id, frame_num, save_path=save_path, show=False
            )

            if not success:
                print(f"Failed to visualize frame {frame_num}")

            print()

        print("=" * 70)
        print("Visualization complete!")
        print("=" * 70)

    def compare_drones(self, frame_num, drone_ids=None, save_path=None):
        """
        Compare the same frame across multiple drones.

        Args:
            frame_num: Frame number to compare
            drone_ids: List of drone IDs to compare (default: [1,2,3,4])
            save_path: Optional path to save the comparison grid
        """
        if drone_ids is None:
            drone_ids = [1, 2, 3, 4]

        print("=" * 70)
        print(f"Comparing Frame {frame_num:04d} Across Drones")
        print("=" * 70)
        print()

        # Load annotations once
        annotations = self.load_annotations(frame_num)
        if annotations is None:
            return False

        # Create subplot grid
        n_drones = len(drone_ids)
        cols = 2
        rows = (n_drones + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
        if n_drones == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, drone_id in enumerate(drone_ids):
            # Load image
            img = self.load_image(drone_id, frame_num)
            if img is None:
                continue

            # Load bboxes (from YOLO labels or JSON)
            visible_persons = self.load_bboxes_for_frame(drone_id, frame_num)

            # Draw bboxes
            img_annotated = self.draw_bboxes(img, visible_persons)

            # Convert to RGB
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

            # Display
            axes[idx].imshow(img_rgb)
            axes[idx].axis("off")
            axes[idx].set_title(
                f"Drone {drone_id} ({len(visible_persons)} persons)",
                fontsize=12,
                fontweight="bold",
            )

            print(f"Drone {drone_id}: {len(visible_persons)} persons visible")

        # Hide unused subplots
        for idx in range(n_drones, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Frame {frame_num:04d} - Multi-Camera Comparison",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Saved comparison to: {save_path}")

        plt.show()

        print()
        print("=" * 70)
        print("Comparison complete!")
        print("=" * 70)

        return True

    def stream_frames(self, drone_id, start_frame=0, end_frame=999):
        """
        Interactive frame viewer - press SPACE to go to next frame, Q to quit.

        Args:
            drone_id: Drone ID (1-8)
            start_frame: Starting frame number (default: 0)
            end_frame: Ending frame number (default: 999)
        """
        print("=" * 70)
        print(f"Interactive Frame Viewer - Drone {drone_id}")
        print("=" * 70)
        print(f"Frames: {start_frame} to {end_frame}")
        print()
        print("Controls:")
        print("  SPACE - Next frame")
        print("  B - Previous frame")
        print("  Q - Quit")
        print("=" * 70)
        print()

        current_frame = start_frame
        running = True

        while running and current_frame <= end_frame:
            # Load image
            img = self.load_image(drone_id, current_frame)
            if img is None:
                print(f"Frame {current_frame:04d} not found, skipping...")
                current_frame += 1
                continue

            # Load annotations
            annotations = self.load_annotations(current_frame)
            if annotations is None:
                print(
                    f"Annotations for frame {current_frame:04d} not found, showing image only..."
                )
                img_annotated = img
                visible_persons = []
            else:
                # Load bboxes (from YOLO labels or JSON)
                visible_persons = self.load_bboxes_for_frame(drone_id, current_frame)

                # Draw bboxes
                img_annotated = self.draw_bboxes(img, visible_persons)

            # Convert to RGB for matplotlib
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.imshow(img_rgb)
            ax.axis("off")
            ax.set_title(
                f"Drone {drone_id} | Frame {current_frame:04d} | Persons: {len(visible_persons)}\n"
                f"SPACE=Next | B=Back | Q=Quit",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            # Handle keyboard input
            user_action = {"action": None}

            def on_key(event):
                if event.key == " " or event.key == "right":
                    user_action["action"] = "next"
                    plt.close(fig)
                elif event.key == "b" or event.key == "left":
                    user_action["action"] = "back"
                    plt.close(fig)
                elif event.key in ["q", "Q"]:
                    user_action["action"] = "quit"
                    plt.close(fig)

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()

            # Process action
            if user_action["action"] == "next":
                print(f"Frame {current_frame:04d} -> {current_frame + 1:04d}")
                current_frame += 1
            elif user_action["action"] == "back":
                if current_frame > start_frame:
                    print(f"Frame {current_frame:04d} -> {current_frame - 1:04d}")
                    current_frame -= 1
                else:
                    print(f"Already at first frame ({start_frame:04d})")
            elif user_action["action"] == "quit":
                print("\nQuitting viewer...")
                running = False
            else:
                # Window closed without key press
                print("\nWindow closed, quitting...")
                running = False

            plt.close("all")

        print()
        print("=" * 70)
        print("Viewer closed!")
        print("=" * 70)


def main():
    """Main entry point with example usage."""
    # Configuration - script is in MATRIX_yolo_format directory
    DATASET_ROOT = Path(__file__).parent  # MATRIX_yolo_format directory

    # Create visualizer
    visualizer = MATRIXAnnotationVisualizer(DATASET_ROOT)

    print()
    print("=" * 70)
    print("MATRIX Dataset Annotation Visualizer")
    print("=" * 70)
    print()

    # Example 1: Visualize a single frame
    print("Example 1: Single frame visualization")
    print("-" * 70)
    visualizer.visualize(
        drone_id=1,
        frame_num=0,
        save_path="visualization_output/drone1_frame0000.png",
        show=True,
    )

    # Example 2: Compare across multiple drones
    print("\nExample 2: Multi-camera comparison")
    print("-" * 70)
    visualizer.compare_drones(
        frame_num=100,
        drone_ids=[1, 2, 3, 4],
        save_path="visualization_output/comparison_frame0100.png",
    )

    # Example 3: Visualize multiple frames (save only, no display)
    print("\nExample 3: Batch visualization")
    print("-" * 70)
    visualizer.visualize_multiple(
        drone_id=1,
        frame_nums=[0, 100, 200, 300, 400, 500],
        save_dir="visualization_output/batch",
    )


if __name__ == "__main__":
    import sys

    # Simple CLI for quick visualization
    if len(sys.argv) == 3:
        # Usage: python visualize_annotations.py <drone_id> <frame_num>
        try:
            drone_id = int(sys.argv[1])
            frame_num = int(sys.argv[2])

            DATASET_ROOT = Path(__file__).parent
            visualizer = MATRIXAnnotationVisualizer(DATASET_ROOT)
            visualizer.visualize(drone_id, frame_num, show=True)

        except ValueError:
            print("Usage: python visualize_annotations.py <drone_id> <frame_num>")
            print("Example: python visualize_annotations.py 1 0")
            sys.exit(1)

    elif len(sys.argv) == 2:
        # Usage: python visualize_annotations.py <drone_id> (interactive viewer)
        try:
            drone_id = int(sys.argv[1])

            DATASET_ROOT = Path(__file__).parent
            visualizer = MATRIXAnnotationVisualizer(DATASET_ROOT)

            print()
            start_frame = int(input("Start frame (0-999, default 0): ") or "0")
            end_frame = int(input("End frame (0-999, default 999): ") or "999")
            print()

            visualizer.stream_frames(drone_id, start_frame, end_frame)

        except ValueError:
            print("Usage: python visualize_annotations.py <drone_id>")
            print("Example: python visualize_annotations.py 1")
            sys.exit(1)

    elif len(sys.argv) == 1:
        # No arguments - run examples
        main()

    else:
        print("Usage:")
        print("  python visualize_annotations.py                    # Run examples")
        print(
            "  python visualize_annotations.py <drone>            # Interactive viewer"
        )
        print("  python visualize_annotations.py <drone> <frame>    # Quick view")
        print()
        print("Examples:")
        print("  python visualize_annotations.py 1        # Stream Drone 1 frames")
        print("  python visualize_annotations.py 1 0      # View Drone 1, Frame 0")
        print("  python visualize_annotations.py 5 500    # View Drone 5, Frame 500")
        sys.exit(1)
