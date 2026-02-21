"""
Feature Extraction Stage

Extracts appearance descriptors from person detections for cross-camera matching.

Components:
- models: Data structures (PersonFeatures, FrameFeatures)
- wch_extractor: Weighted Color Histogram feature extraction

Usage:
    from features import WCHExtractor, PersonFeatures, FrameFeatures

    extractor = WCHExtractor(config.features)

    frame_features = extractor.extract_frame(frame, detections)

    for pf in frame_features.features:
        logger.info("Extracted WCH for person at %s", pf.bbox_center)
"""

from algorithm.features.models import PersonFeatures, FrameFeatures
from algorithm.features.wch_extractor import WCHExtractor

__all__ = ["WCHExtractor", "PersonFeatures", "FrameFeatures"]
