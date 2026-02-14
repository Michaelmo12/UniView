"""
Weighted Color Histogram (WCH) Feature Extractor

Extracts 96-dimensional appearance descriptors from person detections using
HSV color histograms with upper/lower body region weighting.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from detection.models import Detection, DetectionSet
from ingestion.models import DroneFrame
from features.models import PersonFeatures, FrameFeatures
from config.settings import FeatureConfig

logger = logging.getLogger(__name__)


class WCHExtractor:
    """
    Weighted Color Histogram extractor for person appearance matching.

    Algorithm:
    1. Crop person region from frame using bounding box
    2. Resize to fixed dimensions for consistent processing
    3. Convert to HSV color space (robust to lighting changes)
    4. Split vertically into upper (torso) and lower (legs) regions
    5. Compute separate H, S, V histograms for each region and concatenate
    6. Weight regions (upper 60%, lower 40%) and concatenate
    7. L2-normalize for cosine similarity matching

    Output: 96-dimensional vector (16 bins × 3 channels × 2 regions)
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize WCH extractor with configuration.

        Args:
            config: FeatureConfig with histogram and crop parameters
        """
        self.config = config
        logger.info(
            "WCHExtractor initialized: bins=%d, upper_weight=%.1f, lower_weight=%.1f",
            config.bins_per_channel,
            config.upper_weight,
            config.lower_weight,
        )

    def extract_frame(
        self, frame: DroneFrame, detections: DetectionSet
    ) -> FrameFeatures:
        """
        Extract WCH features for all detections in a frame.

        Args:
            frame: DroneFrame with image and calibration
            detections: DetectionSet with bounding boxes

        Returns:
            FrameFeatures containing successfully extracted features
        """
        start_time = time.perf_counter()
        features_list = []

        for detection in detections.detections:
            try:
                wch_vector = self._extract_single(frame.frame, detection)
                if wch_vector is not None:
                    # Set feature vector on detection for downstream stages
                    detection.features = wch_vector

                    # Build PersonFeatures object
                    person_features = PersonFeatures(
                        drone_id=detection.drone_id,
                        frame_num=detection.frame_num,
                        local_id=detection.local_id,
                        wch=wch_vector,
                        bbox_center=detection.bbox.center,
                        projection_matrix=frame.calibration.projection_matrix,
                        confidence=detection.confidence,
                    )
                    features_list.append(person_features)
                else:
                    # Extraction failed (crop too small), set features to None
                    detection.features = None
                    logger.debug(
                        "Skipping detection (crop too small): drone=%d, frame=%d, local_id=%d",
                        detection.drone_id,
                        detection.frame_num,
                        detection.local_id,
                    )
            except Exception as e:
                # Log warning and continue with features=None
                logger.warning(
                    "Feature extraction failed for detection drone=%d, frame=%d, local_id=%d: %s",
                    detection.drone_id,
                    detection.frame_num,
                    detection.local_id,
                    e,
                )
                detection.features = None

        elapsed = time.perf_counter() - start_time
        logger.debug(
            "Extracted features for %d/%d detections in %.1fms (drone=%d, frame=%d)",
            len(features_list),
            len(detections.detections),
            elapsed * 1000,
            frame.drone_id,
            frame.frame_num,
        )

        return FrameFeatures(
            drone_id=frame.drone_id, frame_num=frame.frame_num, features=features_list
        )

    def _extract_single(
        self, image: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """
        Extract WCH for a single detection.

        Args:
            image: BGR image (H, W, 3)
            detection: Detection with bounding box

        Returns:
            96-dimensional L2-normalized WCH vector, or None if crop too small
        """
        # 1. Clip bbox to image bounds
        clipped = detection.bbox.clip(image.shape[1], image.shape[0])

        # 2. Convert float coords to int for array slicing
        x1, y1, x2, y2 = int(clipped.x1), int(clipped.y1), int(clipped.x2), int(clipped.y2)

        # 3. Check minimum size
        width = x2 - x1
        height = y2 - y1
        if width < self.config.min_crop_width or height < self.config.min_crop_height:
            return None

        # 4. Crop person region
        crop = image[y1:y2, x1:x2]

        # 5. Resize to fixed dimensions
        crop_resized = cv2.resize(
            crop,
            (self.config.crop_resize_width, self.config.crop_resize_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # 6. Convert BGR to HSV
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        # 7. Split at vertical midpoint (upper = torso, lower = legs)
        mid = self.config.crop_resize_height // 2
        upper = hsv[:mid, :, :]
        lower = hsv[mid:, :, :]

        # 8. Compute 3 separate 1D histograms for each region and concatenate
        upper_hist = self._compute_region_histogram(upper)
        lower_hist = self._compute_region_histogram(lower)

        # 9. Weight and concatenate regions
        wch = np.concatenate(
            [upper_hist * self.config.upper_weight, lower_hist * self.config.lower_weight]
        )

        # 10. L2 normalize
        norm = np.linalg.norm(wch)
        if norm > 1e-10:
            wch = wch / norm
        else:
            # Degenerate case (uniform black crop) - return zero vector
            wch = wch / 1e-10

        return wch.astype(np.float64)

    def _compute_region_histogram(self, region: np.ndarray) -> np.ndarray:
        """
        Compute concatenated H-S-V histograms for a region.

        Args:
            region: HSV image region (H, W, 3)

        Returns:
            Concatenated histogram (48,) = [H_hist(16) | S_hist(16) | V_hist(16)]
        """
        # H channel: range [0, 180) in OpenCV
        h_hist = np.histogram(
            region[:, :, 0].ravel(),
            bins=self.config.bins_per_channel,
            range=(0, 180),
        )[0].astype(np.float64)

        # S channel: range [0, 256)
        s_hist = np.histogram(
            region[:, :, 1].ravel(),
            bins=self.config.bins_per_channel,
            range=(0, 256),
        )[0].astype(np.float64)

        # V channel: range [0, 256)
        v_hist = np.histogram(
            region[:, :, 2].ravel(),
            bins=self.config.bins_per_channel,
            range=(0, 256),
        )[0].astype(np.float64)

        return np.concatenate([h_hist, s_hist, v_hist])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized histograms.

        Since both vectors are L2-normalized, cosine similarity equals dot product.

        Args:
            a: First feature vector (96,)
            b: Second feature vector (96,)

        Returns:
            Similarity in [0, 1], higher = more similar
        """
        return float(np.clip(np.dot(a, b), 0.0, 1.0))
