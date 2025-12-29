"""
Configuration module for Face Recognition project.
Handles environment detection and path configuration.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Environment(Enum):
    """Enumeration of supported environments."""

    LOCAL = "local"
    KAGGLE = "kaggle"


@dataclass
class PathConfig:
    """Configuration for data and model paths."""

    data_path: str
    pos_path: str
    neg_path: str
    anc_path: str
    model_path: str
    checkpoint_dir: str
    application_data_path: str
    verification_images_path: str
    input_image_path: str


class Config:
    """Main configuration class for the project."""

    # Model parameters
    IMAGE_SIZE = (100, 100)
    INPUT_SHAPE = (100, 100, 3)
    EMBEDDING_SIZE = 4096

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DEFAULT_EPOCHS = 50
    CHECKPOINT_FREQUENCY = 10
    MAX_SAMPLES = 3000
    BUFFER_SIZE = 10000
    TRAIN_SPLIT = 0.7

    # Verification parameters
    DEFAULT_DETECTION_THRESHOLD = 0.5
    DEFAULT_VERIFICATION_THRESHOLD = 0.5

    # Camera settings
    DEFAULT_CAMERA_INDEX = 0
    CAPTURE_SIZE = (250, 250)

    def __init__(self, environment: Optional[Environment] = None):
        """
        Initialize configuration based on environment.

        Args:
            environment: The runtime environment (local or kaggle).
                        If None, auto-detects based on directory structure.
        """
        self.environment = environment or self._detect_environment()
        self.paths = self._setup_paths()

    def _detect_environment(self) -> Environment:
        """Auto-detect the runtime environment."""
        if os.path.exists("/kaggle/working"):
            return Environment.KAGGLE
        return Environment.LOCAL

    def _setup_paths(self) -> PathConfig:
        """Setup paths based on the detected environment."""
        if self.environment == Environment.KAGGLE:
            return self._get_kaggle_paths()
        return self._get_local_paths()

    def _get_kaggle_paths(self) -> PathConfig:
        """Get path configuration for Kaggle environment."""
        data_path = "/kaggle/input/dataset/data"
        working_path = "/kaggle/working"

        return PathConfig(
            data_path=data_path,
            pos_path=os.path.join(data_path, "positive"),
            neg_path=os.path.join(data_path, "negative"),
            anc_path=os.path.join(data_path, "anchor"),
            model_path=os.path.join(working_path, "models", "siamesemodelv2.keras"),
            checkpoint_dir=os.path.join(working_path, "models", "checkpoints"),
            application_data_path=os.path.join(working_path, "application_data"),
            verification_images_path=os.path.join(
                working_path, "application_data", "verification_images"
            ),
            input_image_path=os.path.join(
                working_path, "application_data", "input_image"
            ),
        )

    def _get_local_paths(self) -> PathConfig:
        """Get path configuration for local environment."""
        # Get the directory where this config file is located
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data")

        return PathConfig(
            data_path=data_path,
            pos_path=os.path.join(data_path, "positive"),
            neg_path=os.path.join(data_path, "negative"),
            anc_path=os.path.join(data_path, "anchor"),
            model_path=os.path.join(base_dir, "models", "siamesemodelv2.keras"),
            checkpoint_dir=os.path.join(base_dir, "models", "checkpoints"),
            application_data_path=os.path.join(base_dir, "application_data"),
            verification_images_path=os.path.join(
                base_dir, "application_data", "verification_images"
            ),
            input_image_path=os.path.join(base_dir, "application_data", "input_image"),
        )

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths.data_path,
            self.paths.pos_path,
            self.paths.neg_path,
            self.paths.anc_path,
            self.paths.checkpoint_dir,
            self.paths.application_data_path,
            self.paths.verification_images_path,
            self.paths.input_image_path,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def print_config(self) -> None:
        """Print current configuration."""
        print(f"\n{'='*50}")
        print(f"Face Recognition Configuration")
        print(f"{'='*50}")
        print(f"Environment: {self.environment.value}")
        print(f"\nPaths:")
        print(f"  Data Path: {self.paths.data_path}")
        print(f"  Positive Path: {self.paths.pos_path}")
        print(f"  Negative Path: {self.paths.neg_path}")
        print(f"  Anchor Path: {self.paths.anc_path}")
        print(f"  Model Path: {self.paths.model_path}")
        print(f"  Checkpoint Dir: {self.paths.checkpoint_dir}")
        print(f"{'='*50}\n")
