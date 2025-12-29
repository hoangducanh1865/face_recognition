#!/usr/bin/env python3
"""
Face Recognition - Siamese Network
Main entry point for the application.

Usage:
    python -m src.main [OPTIONS]

Options:
    --mode          : 'train', 'verify', 'collect', or 'evaluate'
    --environment   : 'local' or 'kaggle'
    --use-checkpoint: Use checkpoint instead of trained model
    --epochs        : Number of training epochs
    --camera        : Camera index for webcam
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, Environment
from src.model import SiameseModel
from src.data import DataLoader, DataAugmentor
from src.trainer import Trainer
from src.verifier import FaceVerifier, RealtimeVerifier
from src.utils import (
    setup_gpu,
    print_tensorflow_info,
    list_checkpoints,
    plot_training_history,
)


class FaceRecognitionApp:
    """Main application class for Face Recognition."""

    def __init__(self, environment: str = "local", use_checkpoint: bool = False):
        """
        Initialize the application.

        Args:
            environment: 'local' or 'kaggle'.
            use_checkpoint: Whether to load from checkpoint instead of model file.
        """
        # Setup GPU
        setup_gpu()
        print_tensorflow_info()

        # Initialize configuration
        env = Environment.KAGGLE if environment == "kaggle" else Environment.LOCAL
        self.config = Config(environment=env)
        self.config.print_config()

        # Initialize model
        self.model = SiameseModel(self.config)
        self.use_checkpoint = use_checkpoint

        # Load model or checkpoint if available
        self._load_model()

    def _load_model(self) -> None:
        """Load model from file or checkpoint."""
        if self.use_checkpoint:
            # List available checkpoints
            checkpoints = list_checkpoints(self.config.paths.checkpoint_dir)

            if checkpoints:
                # Create trainer and restore checkpoint
                trainer = Trainer(self.model, self.config)
                trainer.restore_checkpoint()
        else:
            # Try to load saved model
            try:
                self.model.load()
                print("Pre-trained model loaded successfully!")
            except FileNotFoundError:
                print(
                    "No pre-trained model found. Model initialized with random weights."
                )

    def train(self, epochs: int = 50, augment_data: bool = False) -> dict:
        """
        Train the model.

        Args:
            epochs: Number of training epochs.
            augment_data: Whether to augment training data.

        Returns:
            Training history.
        """
        print("\n" + "=" * 50)
        print("Training Mode")
        print("=" * 50)

        # Create directories if needed
        self.config.create_directories()

        # Augment data if requested
        if augment_data:
            print("\nAugmenting data...")
            augmentor = DataAugmentor()
            augmentor.augment_directory(self.config.paths.anc_path)
            augmentor.augment_directory(self.config.paths.pos_path)

        # Load data
        print("\nLoading dataset...")
        data_loader = DataLoader(self.config)

        # Check data availability
        info = data_loader.get_dataset_info()
        print(f"\nDataset info:")
        for name, count in info.items():
            print(f"  {name}: {count} images")

        if any(count == 0 for count in info.values()):
            print("\nWarning: Some data directories are empty!")
            print("Please collect images first using --mode collect")
            return {}

        train_data, test_data = data_loader.load_dataset()

        # Create trainer
        trainer = Trainer(self.model, self.config)

        # Optionally restore from checkpoint
        if self.use_checkpoint:
            trainer.restore_checkpoint()

        # Train
        history = trainer.train(train_data, epochs=epochs)

        # Evaluate
        print("\n" + "=" * 50)
        print("Evaluation")
        print("=" * 50)
        trainer.evaluate(test_data)

        # Save model
        self.model.save()

        return history

    def verify(
        self,
        camera_index: int = 0,
        detection_threshold: float = 0.5,
        verification_threshold: float = 0.5,
    ) -> None:
        """
        Run real-time face verification.

        Args:
            camera_index: Webcam index.
            detection_threshold: Threshold for positive detection.
            verification_threshold: Threshold for verification.
        """
        print("\n" + "=" * 50)
        print("Verification Mode")
        print("=" * 50)

        # Create verifier
        verifier = FaceVerifier(self.model, self.config)

        # Check for verification images
        count = verifier.get_verification_image_count()
        print(f"\nVerification images found: {count}")

        if count == 0:
            print("\nNo verification images found!")
            print("Please add images to:", self.config.paths.verification_images_path)
            return

        # Create real-time verifier
        rt_verifier = RealtimeVerifier(verifier, camera_index, self.config)

        # Run verification loop
        rt_verifier.run_verification_loop(detection_threshold, verification_threshold)

    def collect(self, camera_index: int = 0) -> None:
        """
        Collect training images using webcam.

        Args:
            camera_index: Webcam index.
        """
        print("\n" + "=" * 50)
        print("Image Collection Mode")
        print("=" * 50)

        # Create directories
        self.config.create_directories()

        # Create verifier for camera access
        verifier = FaceVerifier(self.model, self.config)
        rt_verifier = RealtimeVerifier(verifier, camera_index, self.config)

        # Collect images
        rt_verifier.collect_images(
            self.config.paths.anc_path, self.config.paths.pos_path
        )

    def evaluate(self) -> dict:
        """
        Evaluate the model on test data.

        Returns:
            Evaluation metrics.
        """
        print("\n" + "=" * 50)
        print("Evaluation Mode")
        print("=" * 50)

        # Load data
        data_loader = DataLoader(self.config)
        _, test_data = data_loader.load_dataset()

        # Create trainer for evaluation
        trainer = Trainer(self.model, self.config)

        # Evaluate
        results = trainer.evaluate(test_data)

        return results

    def show_model_info(self) -> None:
        """Display model architecture information."""
        self.model.summary()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Face Recognition using Siamese Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on local machine
    python -m src.main --mode train --environment local --epochs 50
    
    # Train on Kaggle with checkpoint
    python -m src.main --mode train --environment kaggle --use-checkpoint
    
    # Run verification
    python -m src.main --mode verify --camera 0
    
    # Collect training images
    python -m src.main --mode collect --camera 0
    
    # Evaluate model
    python -m src.main --mode evaluate
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "verify", "collect", "evaluate", "info"],
        default="info",
        help="Operation mode (default: info)",
    )

    parser.add_argument(
        "--environment",
        type=str,
        choices=["local", "kaggle"],
        default="local",
        help="Runtime environment (default: local)",
    )

    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Load from checkpoint instead of model file",
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index for webcam (default: 0)"
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)",
    )

    parser.add_argument(
        "--verification-threshold",
        type=float,
        default=0.5,
        help="Verification threshold (default: 0.5)",
    )

    parser.add_argument(
        "--augment", action="store_true", help="Augment training data before training"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize application
    app = FaceRecognitionApp(
        environment=args.environment, use_checkpoint=args.use_checkpoint
    )

    # Execute based on mode
    if args.mode == "train":
        history = app.train(epochs=args.epochs, augment_data=args.augment)
        if history:
            try:
                plot_training_history(history)
            except Exception:
                pass  # Skip plotting if display not available

    elif args.mode == "verify":
        app.verify(
            camera_index=args.camera,
            detection_threshold=args.detection_threshold,
            verification_threshold=args.verification_threshold,
        )

    elif args.mode == "collect":
        app.collect(camera_index=args.camera)

    elif args.mode == "evaluate":
        app.evaluate()

    elif args.mode == "info":
        app.show_model_info()


if __name__ == "__main__":
    main()
