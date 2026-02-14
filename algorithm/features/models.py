from dataclasses import dataclass, field
import numpy as np


@dataclass
class PersonFeatures:
    """
    Extracted features for one detected person.
    """

    drone_id: int
    frame_num: int
    local_id: int
    wch: np.ndarray  # 96-dimensional L2-normalized WCH descriptor
    bbox_center: tuple[float, float]
    projection_matrix: np.ndarray  # (3, 4)
    confidence: float  # Detection confidence (passed through for downstream filtering)


@dataclass
class FrameFeatures:
    """
    All extracted features for one drone frame.
    """

    drone_id: int
    frame_num: int
    features: list[PersonFeatures] = field(default_factory=list)

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def is_empty(self) -> bool:
        return len(self.features) == 0
