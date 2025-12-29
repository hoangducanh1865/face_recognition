"""
Face verification module for real-time face recognition.
"""

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf

from .config import Config
from .model import SiameseModel
from .data import DataPreprocessor


class FaceVerifier:
    """Handles face verification using the trained Siamese Network."""

    def __init__(self, model: SiameseModel, config: Optional[Config] = None):
        """
        Initialize the verifier.

        Args:
            model: Trained Siamese model.
            config: Configuration object.
        """
        self.model = model
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)

        # Ensure directories exist
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories for verification."""
        os.makedirs(self.config.paths.verification_images_path, exist_ok=True)
        os.makedirs(self.config.paths.input_image_path, exist_ok=True)

    def verify(
        self,
        detection_threshold: Optional[float] = None,
        verification_threshold: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], bool]:
        """
        Verify face against stored verification images.

        Args:
            detection_threshold: Threshold above which a prediction is positive.
            verification_threshold: Proportion of positives needed for verification.

        Returns:
            Tuple of (prediction_results, is_verified).
        """
        detection_threshold = (
            detection_threshold or self.config.DEFAULT_DETECTION_THRESHOLD
        )
        verification_threshold = (
            verification_threshold or self.config.DEFAULT_VERIFICATION_THRESHOLD
        )

        verification_path = self.config.paths.verification_images_path
        input_image_path = os.path.join(
            self.config.paths.input_image_path, "input_image.jpg"
        )

        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        verification_images = os.listdir(verification_path)
        if not verification_images:
            raise ValueError("No verification images found!")

        results = []

        # Filter valid image files
        valid_images = [
            img
            for img in verification_images
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(
            f"Comparing against {len(valid_images)} verification images...", flush=True
        )

        for idx, image_name in enumerate(valid_images):
            # Preprocess images
            input_img = self.preprocessor.preprocess(input_image_path)
            validation_img = self.preprocessor.preprocess(
                os.path.join(verification_path, image_name)
            )

            # Make prediction
            result = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1)),
                verbose=0,  # Suppress prediction output
            )
            results.append(result)

            # Show progress every 10 images
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(valid_images)}...", flush=True)

        # Calculate verification
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(valid_images) if valid_images else 0
        is_verified = verification > verification_threshold

        return results, is_verified

    def add_verification_image(self, image: np.ndarray, name: str) -> str:
        """
        Add a new verification image.

        Args:
            image: Image as numpy array (BGR format from OpenCV).
            name: Name for the image file.

        Returns:
            Path to the saved image.
        """
        path = os.path.join(self.config.paths.verification_images_path, f"{name}.jpg")
        cv2.imwrite(path, image)
        print(f"Verification image saved: {path}")
        return path

    def set_input_image(self, image: np.ndarray) -> str:
        """
        Set the input image for verification.

        Args:
            image: Image as numpy array (BGR format from OpenCV).

        Returns:
            Path to the saved image.
        """
        path = os.path.join(self.config.paths.input_image_path, "input_image.jpg")
        cv2.imwrite(path, image)
        return path

    def get_verification_image_count(self) -> int:
        """Get the number of stored verification images."""
        path = self.config.paths.verification_images_path
        if not os.path.exists(path):
            return 0
        return len(
            [f for f in os.listdir(path) if f.endswith((".jpg", ".jpeg", ".png"))]
        )


class RealtimeVerifier:
    """Real-time face verification using webcam."""

    def __init__(
        self,
        verifier: FaceVerifier,
        camera_index: Optional[int] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the real-time verifier.

        Args:
            verifier: FaceVerifier instance.
            camera_index: Webcam index.
            config: Configuration object.
        """
        self.verifier = verifier
        self.config = config or Config()
        self.camera_index = camera_index or self.config.DEFAULT_CAMERA_INDEX
        self.capture_size = self.config.CAPTURE_SIZE

    def run_verification_loop(
        self,
        detection_threshold: Optional[float] = None,
        verification_threshold: Optional[float] = None,
    ) -> None:
        """
        Run the real-time verification loop.

        Controls:
            - 'v': Capture and verify face
            - 'q': Quit

        Args:
            detection_threshold: Threshold for positive detection.
            verification_threshold: Threshold for verification.
        """
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")

        print("\n" + "=" * 50)
        print("Real-time Face Verification")
        print("=" * 50)
        print("Controls:")
        print("  'v' - Capture and verify face")
        print("  'q' - Quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize frame
                frame_resized = cv2.resize(frame, self.capture_size)

                # Add text overlay with instructions
                display_frame = frame_resized.copy()
                cv2.putText(
                    display_frame,
                    "Press 'v' to verify, 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Display frame
                cv2.imshow("Face Verification", display_frame)

                key = cv2.waitKey(1) & 0xFF

                # Verify on 'v' press
                if key == ord("v"):
                    print("\nCapturing image for verification...", flush=True)
                    self.verifier.set_input_image(frame_resized)

                    try:
                        results, verified = self.verifier.verify(
                            detection_threshold, verification_threshold
                        )

                        status = "VERIFIED ✓" if verified else "NOT VERIFIED ✗"
                        confidence = (
                            np.mean([r[0][0] for r in results]) if results else 0
                        )

                        print(f"\n{'='*30}")
                        print(f"Result: {status}")
                        print(f"Confidence: {confidence:.2%}")
                        print(f"{'='*30}\n", flush=True)
                    except Exception as e:
                        print(f"Verification error: {e}", flush=True)

                # Quit on 'q' press
                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nVerification session ended.")

    def collect_images(
        self, anchor_path: str, positive_path: str, mode: str = "both"
    ) -> None:
        """
        Collect anchor and positive images for training.

        Controls:
            - 'a': Save as anchor image
            - 'p': Save as positive image
            - 'q': Quit

        Args:
            anchor_path: Path to save anchor images.
            positive_path: Path to save positive images.
            mode: 'anchor', 'positive', or 'both'.
        """
        import uuid

        os.makedirs(anchor_path, exist_ok=True)
        os.makedirs(positive_path, exist_ok=True)

        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")

        print("\n" + "=" * 50)
        print("Image Collection Mode")
        print("=" * 50)
        print("Controls:")
        if mode in ("anchor", "both"):
            print("  'a' - Save as anchor image")
        if mode in ("positive", "both"):
            print("  'p' - Save as positive image")
        print("  'q' - Quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                frame_resized = cv2.resize(frame, self.capture_size)
                cv2.imshow("Image Collection", frame_resized)

                key = cv2.waitKey(1) & 0xFF

                # Save anchor
                if key == ord("a") and mode in ("anchor", "both"):
                    path = os.path.join(anchor_path, f"{uuid.uuid1()}.jpg")
                    cv2.imwrite(path, frame_resized)
                    print(f"Saved anchor: {path}")

                # Save positive
                if key == ord("p") and mode in ("positive", "both"):
                    path = os.path.join(positive_path, f"{uuid.uuid1()}.jpg")
                    cv2.imwrite(path, frame_resized)
                    print(f"Saved positive: {path}")

                # Quit
                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nImage collection ended.")
