from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network configuration for ENet (UDP-based) receivers."""

    base_port: int = 15000  # Drone 1 on port 15000, drone 2 on 15001, etc.
    host: str = "127.0.0.1"
    recv_timeout: float = 10.0  # ENet receive timeout (seconds)
    reconnect_delay: float = 1.0  # Delay before reconnecting after disconnect


@dataclass
class IngestionConfig:
    """Ingestion stage configuration."""

    drone_ids: list = None  # Drone IDs to connect to (e.g. [3,4,6,7])
    sync_timeout: float = 1.0  # Frame synchronization timeout (seconds)
    max_buffer_size: int = 10  # Max frames buffered per synchronizer

    def __post_init__(self):
        if self.drone_ids is None:
            self.drone_ids = [3, 4, 6, 7]


@dataclass
class DetectionConfig:
    """Detection stage configuration."""

    weights_file: str = "nano/best_openvino_model"  # OpenVINO FP32 (nano)
    conf_threshold: float = 0.5  # Minimum detection confidence
    iou_threshold: float = 0.45  # NMS IoU threshold
    person_class_id: int = 0  # Class ID for person
    imgsz: int = 640  # YOLO input size
    device: str = "cpu"  # "cuda" or "cpu"


@dataclass
class FeatureConfig:
    """Feature extraction stage configuration."""

    bins_per_channel: int = 16  # Histogram bins per HSV channel (16 x 3 x 2 = 96 dims)
    upper_weight: float = 0.6  # Torso region weight (more distinctive clothing)
    lower_weight: float = 0.4  # Legs region weight
    min_crop_width: int = 8  # Minimum crop width (below this, histograms too noisy)
    min_crop_height: int = 16  # Minimum crop height in pixels
    crop_resize_height: int = 128  # Resize crops to fixed height for consistent split
    crop_resize_width: int = 64  # Resize crops to fixed width


@dataclass
class FusionConfig:
    """Cross-camera fusion stage configuration."""

    epipolar_threshold: float = (
        2.0  # Max point-to-epiline distance (pixels) for geometric match
    )
    appearance_threshold: float = 0.55  # Min WCH cosine similarity for appearance match
    min_cameras: int = 2  # Minimum cameras that must observe a person for valid match


@dataclass
class ReconstructionConfig:
    """3D reconstruction stage configuration."""

    max_reprojection_error: float = (
        2.5  # Max avg reprojection error (pixels) to accept triangulation
    )
    dbscan_eps: float = (
        0.5  # DBSCAN epsilon (meters) -- max distance between points in cluster
    )
    dbscan_min_samples: int = (
        2  # DBSCAN min_samples -- minimum 2 points to form cluster (isolated points become noise)
    )
    prune_bad_ratio_threshold: float = (
        0.60  # Robust triangulation: drop a view if it is bad in >= 60% of pairwise tests
    )


@dataclass
class TrackingConfig:
    """Temporal tracking stage configuration."""

    n_init: int = (
        2  # Consecutive hits to confirm (conservative, reduces false positives)
    )
    max_age: int = (
        5  # Max frames coasting before deletion (generous, handles brief occlusions at 2 FPS)
    )
    max_distance: float = (
        3.5  # Max association distance in meters (allows ~1.5m/frame movement)
    )
    process_noise: float = 0.5  # Kalman Q diagonal scale
    measurement_noise: float = (
        0.5  # Kalman R diagonal scale (accounts for triangulation variance ~0.5m)
    )


@dataclass
class OutputConfig:
    """HTTP POST output configuration."""

    gateway_url: str = (
        "http://localhost:8080"  # Base URL for gateway POST /api/internal/push
    )
    jpeg_quality: int = 70  # JPEG compression quality (0-100)


@dataclass
class GeometryConfig:
    """Geometry coordinate-system configuration.

    Controls how 2D image coordinates are mapped before camera-geometry stages
    (epipolar, triangulation, reprojection).
    """

    flip_x_for_geometry: bool = True  # Apply x' = W - x before geometric computations
    image_width_override: int = 0  # 0 means use runtime frame width


class Settings:
    """
    Root settings container with Singleton Pattern.

    Sub-configurations:
    - network: ENet (UDP-based) receiver settings
    - ingestion: Frame synchronization settings
    - detection: YOLO detector settings
    - features: Feature extraction settings
    - fusion: Cross-camera fusion settings
    - reconstruction: Triangulation and clustering settings
    - tracking: Kalman filter and association settings
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize configuration only once.
        Subsequent calls to __init__ are ignored due to _initialized flag.
        """
        if Settings._initialized:
            return

        # Sub-configurations
        self.network = NetworkConfig()
        self.ingestion = IngestionConfig()
        self.detection = DetectionConfig()
        self.features = FeatureConfig()
        self.fusion = FusionConfig()
        self.reconstruction = ReconstructionConfig()
        self.tracking = TrackingConfig()
        self.output = OutputConfig()
        self.geometry = GeometryConfig()

        Settings._initialized = True

    @property
    def base_dir(self) -> Path:
        """Base directory of the algorithm code."""
        return Path(__file__).parent.parent.parent

    @property
    def weights_dir(self) -> Path:
        return self.base_dir / "weights"

    @property
    def weights_path(self) -> Path:
        """Get full path to YOLO weights file."""
        return self.weights_dir / self.detection.weights_file


# Singleton instance - all modules import this same object
# Even if Settings() is called again elsewhere, it returns this same instance
settings = Settings()
