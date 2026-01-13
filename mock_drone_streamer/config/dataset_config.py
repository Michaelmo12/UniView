"""
Dataset Configuration

Configuration for MATRIX dataset paths and parameters.
"""

from pathlib import Path


class DatasetConfig:
    """Dataset configuration constants"""

    # MATRIX dataset root path (relative to project root)
    DATASET_PATH = str(Path(__file__).parent.parent.parent / "MATRIX_30x30" / "MATRIX_30x30")

    # Validate dataset path on import
    @classmethod
    def validate_path(cls):
        """
        Validate that dataset path exists.

        Returns:
            bool: True if dataset exists, False otherwise
        """
        return Path(cls.DATASET_PATH).exists()

    @classmethod
    def get_path(cls):
        """
        Get dataset path as Path object.

        Returns:
            Path: Dataset root path
        """
        return Path(cls.DATASET_PATH)
