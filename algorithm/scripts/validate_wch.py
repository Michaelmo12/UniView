"""
WCH Feature Extraction Validation Script

Tests all Phase 1 success criteria:
- Same-person similarity > 0.7
- Different-person similarity < 0.5
- Latency < 50ms for 8-frame set
- L2 normalization correct
- Edge cases handled without crashes
"""

import sys
import logging
import time
from pathlib import Path
from typing import Tuple

# Add algorithm directory to path for imports
SCRIPT_DIR = Path(__file__).parent
ALGORITHM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALGORITHM_DIR))

import cv2
import numpy as np

from features import WCHExtractor
from config.settings import settings
from detection.models import BoundingBox, Detection, DetectionSet
from ingestion.models import DroneFrame, CameraCalibration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_mock_frame(drone_id: int, frame_num: int, image: np.ndarray) -> DroneFrame:
    """
    Build a DroneFrame with given BGR image and dummy calibration.

    Args:
        drone_id: Drone identifier (1-8)
        frame_num: Frame sequence number
        image: BGR image (H, W, 3)

    Returns:
        DroneFrame with identity R, zero t, standard K
    """
    # Dummy calibration: identity R, zero t, standard focal length
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = 1000.0  # fx, fy
    K[0, 2] = 960.0  # cx
    K[1, 2] = 540.0  # cy

    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

    calibration = CameraCalibration(K=K, R=R, t=t, dist=dist)

    return DroneFrame(
        drone_id=drone_id,
        frame_num=frame_num,
        timestamp=0.0,
        frame=image,
        calibration=calibration,
    )


def make_color_crop(
    height: int, width: int, upper_hsv: Tuple[int, int, int], lower_hsv: Tuple[int, int, int]
) -> np.ndarray:
    """
    Create a BGR image with top half in upper_hsv color and bottom half in lower_hsv color.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        upper_hsv: (H, S, V) for top half
        lower_hsv: (H, S, V) for bottom half

    Returns:
        BGR image (height, width, 3) uint8
    """
    # Convert HSV colors to BGR
    upper_bgr = cv2.cvtColor(
        np.uint8([[list(upper_hsv)]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    lower_bgr = cv2.cvtColor(
        np.uint8([[list(lower_hsv)]]), cv2.COLOR_HSV2BGR
    )[0, 0]

    # Create image with upper and lower halves
    mid = height // 2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:mid, :] = upper_bgr  # Top half
    image[mid:, :] = lower_bgr  # Bottom half

    return image


def make_detection(
    x1: float, y1: float, x2: float, y2: float, drone_id: int = 1, frame_num: int = 0
) -> Detection:
    """
    Shorthand builder for Detection.

    Args:
        x1, y1, x2, y2: Bounding box coordinates
        drone_id: Drone identifier
        frame_num: Frame sequence number

    Returns:
        Detection with class_id=0, confidence=0.9, local_id=0
    """
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return Detection(
        bbox=bbox,
        class_id=0,
        confidence=0.9,
        drone_id=drone_id,
        frame_num=frame_num,
        local_id=0,
    )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_same_person_similarity(extractor: WCHExtractor) -> Tuple[bool, float]:
    """
    Test 1: Same person across different views should have cosine similarity > 0.7.

    Returns:
        (pass, similarity)
    """
    # Person A view 1: blue shirt, dark pants
    # Use random noise to create realistic color variation
    crop1_base = make_color_crop(400, 200, upper_hsv=(120, 200, 180), lower_hsv=(10, 180, 100))
    crop1 = crop1_base.copy()
    # Add small amount of noise to simulate real-world variation
    noise = np.random.randint(-5, 6, crop1.shape, dtype=np.int16)
    crop1 = np.clip(crop1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Person A view 2: same person with similar colors plus noise
    crop2_base = make_color_crop(400, 200, upper_hsv=(120, 200, 180), lower_hsv=(10, 180, 100))
    crop2 = crop2_base.copy()
    noise = np.random.randint(-5, 6, crop2.shape, dtype=np.int16)
    crop2 = np.clip(crop2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place crops in 1080x1920 black frames
    image1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image1[100:500, 100:300] = crop1
    image2[100:500, 100:300] = crop2

    # Create frames and detections
    frame1 = make_mock_frame(1, 0, image1)
    frame2 = make_mock_frame(1, 1, image2)
    det1 = make_detection(100, 100, 300, 500)
    det2 = make_detection(100, 100, 300, 500)

    detections1 = DetectionSet(drone_id=1, frame_num=0, detections=[det1])
    detections2 = DetectionSet(drone_id=1, frame_num=1, detections=[det2])

    # Extract features
    features1 = extractor.extract_frame(frame1, detections1)
    features2 = extractor.extract_frame(frame2, detections2)

    wch1 = features1.features[0].wch
    wch2 = features2.features[0].wch

    # Compute similarity
    similarity = WCHExtractor.cosine_similarity(wch1, wch2)

    return similarity > 0.7, similarity


def test_different_person_similarity(extractor: WCHExtractor) -> Tuple[bool, float]:
    """
    Test 2: Different persons should have cosine similarity < 0.5.

    Returns:
        (pass, similarity)
    """
    # Person A: blue shirt, dark pants
    crop_a = make_color_crop(400, 200, upper_hsv=(120, 200, 180), lower_hsv=(10, 180, 100))
    # Person B: green shirt, yellow pants
    crop_b = make_color_crop(400, 200, upper_hsv=(60, 200, 200), lower_hsv=(30, 220, 220))

    # Place crops in frames
    image_a = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image_b = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image_a[100:500, 100:300] = crop_a
    image_b[100:500, 100:300] = crop_b

    # Create frames and detections
    frame_a = make_mock_frame(1, 0, image_a)
    frame_b = make_mock_frame(2, 0, image_b)
    det_a = make_detection(100, 100, 300, 500, drone_id=1)
    det_b = make_detection(100, 100, 300, 500, drone_id=2)

    detections_a = DetectionSet(drone_id=1, frame_num=0, detections=[det_a])
    detections_b = DetectionSet(drone_id=2, frame_num=0, detections=[det_b])

    # Extract features
    features_a = extractor.extract_frame(frame_a, detections_a)
    features_b = extractor.extract_frame(frame_b, detections_b)

    wch_a = features_a.features[0].wch
    wch_b = features_b.features[0].wch

    # Compute similarity
    similarity = WCHExtractor.cosine_similarity(wch_a, wch_b)

    return similarity < 0.5, similarity


def test_latency(extractor: WCHExtractor) -> Tuple[bool, float]:
    """
    Test 3: Extraction latency for 8 frames with 4 detections each should be < 50ms.

    Returns:
        (pass, median_latency_ms)
    """
    # Create 8 frames with random images
    frames = []
    detection_sets = []

    for drone_id in range(1, 9):
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame = make_mock_frame(drone_id, 0, image)
        frames.append(frame)

        # Create 4 detections at random valid positions
        detections = []
        for i in range(4):
            x1 = np.random.randint(0, 1700)
            y1 = np.random.randint(0, 700)
            x2 = x1 + 150
            y2 = y1 + 300
            det = make_detection(x1, y1, x2, y2, drone_id=drone_id)
            detections.append(det)

        detection_sets.append(
            DetectionSet(drone_id=drone_id, frame_num=0, detections=detections)
        )

    # Run 3 iterations and take median to avoid cold-start outlier
    times = []
    for _ in range(3):
        start = time.perf_counter()
        for frame, det_set in zip(frames, detection_sets):
            extractor.extract_frame(frame, det_set)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    median_time = sorted(times)[1]  # Median of 3 values

    return median_time < 50.0, median_time


def test_l2_normalization(extractor: WCHExtractor) -> Tuple[bool, float]:
    """
    Test 4: Extracted feature vectors should be L2-normalized (norm ≈ 1.0).

    Returns:
        (pass, norm)
    """
    # Create a test image with some color variation
    crop = make_color_crop(400, 200, upper_hsv=(90, 150, 200), lower_hsv=(40, 180, 150))
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[100:500, 100:300] = crop

    frame = make_mock_frame(1, 0, image)
    det = make_detection(100, 100, 300, 500)
    detections = DetectionSet(drone_id=1, frame_num=0, detections=[det])

    # Extract features
    features = extractor.extract_frame(frame, detections)
    wch = features.features[0].wch

    # Check L2 norm
    norm = np.linalg.norm(wch)

    return abs(norm - 1.0) < 0.001, norm


def test_too_small_crop(extractor: WCHExtractor) -> Tuple[bool, str]:
    """
    Test 5: Too-small crop should return FrameFeatures with 0 features, not crash.

    Returns:
        (pass, message)
    """
    # Create a frame with random image
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame = make_mock_frame(1, 0, image)

    # Create a detection with 5x5 pixel bbox (below min_crop_width/height)
    det = make_detection(100, 100, 105, 105)
    detections = DetectionSet(drone_id=1, frame_num=0, detections=[det])

    # Extract features - should handle gracefully
    try:
        features = extractor.extract_frame(frame, detections)
        if features.num_features == 0 and det.features is None:
            return True, "returned 0 features correctly"
        else:
            return False, f"expected 0 features, got {features.num_features}"
    except Exception as e:
        return False, f"crashed with: {e}"


def test_empty_detection_set(extractor: WCHExtractor) -> Tuple[bool, str]:
    """
    Test 6: Empty DetectionSet should return FrameFeatures with 0 features.

    Returns:
        (pass, message)
    """
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame = make_mock_frame(1, 0, image)

    # Empty detection set
    detections = DetectionSet(drone_id=1, frame_num=0, detections=[])

    try:
        features = extractor.extract_frame(frame, detections)
        if features.num_features == 0:
            return True, "0 features"
        else:
            return False, f"expected 0 features, got {features.num_features}"
    except Exception as e:
        return False, f"crashed with: {e}"


def test_edge_bbox(extractor: WCHExtractor) -> Tuple[bool, str]:
    """
    Test 7: Bbox extending beyond image boundary should be handled gracefully.

    Returns:
        (pass, message)
    """
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame = make_mock_frame(1, 0, image)

    # Create detection extending past image edge (x2=2000 on 1920-wide image)
    det = make_detection(1700, 100, 2000, 400)
    detections = DetectionSet(drone_id=1, frame_num=0, detections=[det])

    try:
        features = extractor.extract_frame(frame, detections)
        # Should either clip and extract, or skip if clipped size too small
        return True, "handled gracefully"
    except Exception as e:
        return False, f"crashed with: {e}"


# =============================================================================
# MAIN VALIDATION
# =============================================================================


def main():
    """Run all validation tests and report results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("WCH Feature Extraction Validation")
    logger.info("=" * 60)
    logger.info("")

    # Initialize extractor
    extractor = WCHExtractor(settings.features)

    # Track results
    results = []

    # Test 1: Same-person similarity
    passed, similarity = test_same_person_similarity(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info(
        "[%s] Same-person similarity: %.2f (threshold: >0.7)", status, similarity
    )

    # Test 2: Different-person similarity
    passed, similarity = test_different_person_similarity(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info(
        "[%s] Different-person similarity: %.2f (threshold: <0.5)", status, similarity
    )

    # Test 3: Latency
    passed, latency = test_latency(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info(
        "[%s] Latency (8 frames, 32 detections): %.1fms (threshold: <50ms)",
        status,
        latency,
    )

    # Test 4: L2 normalization
    passed, norm = test_l2_normalization(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] L2 normalization: norm=%.4f", status, norm)

    # Test 5: Too-small crop
    passed, message = test_too_small_crop(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] Too-small crop: %s", status, message)

    # Test 6: Empty detection set
    passed, message = test_empty_detection_set(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] Empty detection set: %s", status, message)

    # Test 7: Edge bbox
    passed, message = test_edge_bbox(extractor)
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    logger.info("[%s] Edge bbox: %s", status, message)

    # Summary
    passed_count = sum(results)
    total_count = len(results)

    logger.info("")
    logger.info("Results: %d/%d passed", passed_count, total_count)
    logger.info("=" * 60)
    logger.info("")

    # Exit with appropriate code
    if passed_count == total_count:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
