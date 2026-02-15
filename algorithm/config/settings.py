from dataclasses import dataclass
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network configuration for TCP receivers."""

    base_port: int = 15000  # Drone 1 on port 15000, drone 2 on 15001, etc.
    host: str = "127.0.0.1"
    recv_timeout: float = 10.0  # Socket receive timeout (seconds)
    reconnect_delay: float = 5.0  # Delay before reconnecting after disconnect


@dataclass
class IngestionConfig:
    """Ingestion stage configuration."""

    num_drones: int = 8  # Expected number of drones
    sync_timeout: float = 0.2  # Frame synchronization timeout (seconds)
    max_buffer_size: int = 100  # Max frames buffered per synchronizer


@dataclass
class DetectionConfig:
    """Detection stage configuration."""

    # weights_file: str = "yolo11s.pt"
    weights_file: str = "best.pt"  # YOLO weights filename (in weights/)
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
    min_crop_width: int = 20  # Minimum crop width (below this, histograms too noisy)
    min_crop_height: int = 40  # Minimum crop height in pixels
    crop_resize_height: int = 128  # Resize crops to fixed height for consistent split
    crop_resize_width: int = 64  # Resize crops to fixed width


@dataclass
class FusionConfig:
    """Cross-camera fusion stage configuration."""

    epipolar_threshold: float = 5.0  # Max point-to-epiline distance (pixels) for geometric match
    appearance_threshold: float = 0.7  # Min WCH cosine similarity for appearance match
    min_cameras: int = 2  # Minimum cameras that must observe a person for valid match


class Settings:
    """
    Root settings container with Singleton Pattern.

    Sub-configurations:
    - network: TCP receiver settings
    - ingestion: Frame synchronization settings
    - detection: YOLO detector settings
    - features: Feature extraction settings
    - fusion: Cross-camera fusion settings
    - reconstruction: (TODO) Triangulation settings
    - tracking: (TODO) Kalman filter settings
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

        Settings._initialized = True

    @property
    def base_dir(self) -> Path:
        """Base directory of the algorithm code."""
        return Path(__file__).parent.parent

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


if __name__ == "__main__":
    """Test: python config/settings.py"""
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Algorithm Configuration")
    logger.info("=" * 60)
    logger.info("")

    # SINGLETON PATTERN TEST
    logger.info("Testing Singleton Pattern:")
    logger.info("  Creating first instance: settings1 = Settings()")
    settings1 = Settings()
    logger.info("  Creating second instance: settings2 = Settings()")
    settings2 = Settings()
    logger.info("  Creating third instance: settings3 = Settings()")
    settings3 = Settings()

    logger.info("  settings1 is settings2: %s", settings1 is settings2)
    logger.info("  settings2 is settings3: %s", settings2 is settings3)
    logger.info("  settings1 is settings: %s", settings1 is settings)
    logger.info("  id(settings1): %s", id(settings1))
    logger.info("  id(settings2): %s", id(settings2))
    logger.info("  id(settings3): %s", id(settings3))

    if settings1 is settings2 is settings3 is settings:
        logger.info("  ✓ SINGLETON PATTERN WORKS! All instances are identical.")
    else:
        logger.error("  ✗ SINGLETON PATTERN FAILED!")

    logger.info("")
    logger.info("Paths:")
    logger.info("  Base directory: %s", settings.base_dir)
    logger.info("  Weights directory: %s", settings.weights_dir)
    logger.info("  Weights path: %s", settings.weights_path)
    logger.info("  Weights exists: %s", settings.weights_path.exists())
    logger.info("")
    logger.info("Network:")
    logger.info("  Base port: %d", settings.network.base_port)
    logger.info("  Host: %s", settings.network.host)
    logger.info("  Recv timeout: %.1fs", settings.network.recv_timeout)
    logger.info("")
    logger.info("Ingestion:")
    logger.info("  Num drones: %d", settings.ingestion.num_drones)
    logger.info("  Sync timeout: %.3fs", settings.ingestion.sync_timeout)
    logger.info("  Max buffer: %d", settings.ingestion.max_buffer_size)
    logger.info("")
    logger.info("Detection:")
    logger.info("  Weights file: %s", settings.detection.weights_file)
    logger.info("  Confidence: %.2f", settings.detection.conf_threshold)
    logger.info("  IoU threshold: %.2f", settings.detection.iou_threshold)
    logger.info("  Device: %s", settings.detection.device)
    logger.info("  Image size: %d", settings.detection.imgsz)
    logger.info("")
    logger.info("=" * 60)
