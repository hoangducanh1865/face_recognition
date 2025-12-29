"""
Data handling module for Face Recognition.
Includes data loading, preprocessing, and augmentation.
"""

import os
import uuid
from typing import Tuple, Optional, List

import cv2
import numpy as np
import tensorflow as tf

from .config import Config


class DataPreprocessor:
    """Handles image preprocessing for the Siamese Network."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.image_size = self.config.IMAGE_SIZE

    def preprocess(self, file_path: str) -> tf.Tensor:
        """
        Preprocess an image for model input.

        Args:
            file_path: Path to the image file.

        Returns:
            Preprocessed image tensor.
        """
        # Read image from file path
        byte_img = tf.io.read_file(file_path)
        # Decode the image
        img = tf.io.decode_jpeg(byte_img)
        # Resize to model input size
        img = tf.image.resize(img, self.image_size)
        # Scale to [0, 1]
        img = img / 255.0
        return img

    def preprocess_twin(
        self, input_img: str, validation_img: str, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Preprocess a pair of images with their label.

        Args:
            input_img: Path to the anchor image.
            validation_img: Path to the validation image.
            label: Ground truth label (1 for same person, 0 for different).

        Returns:
            Tuple of (preprocessed_anchor, preprocessed_validation, label).
        """
        return (self.preprocess(input_img), self.preprocess(validation_img), label)


class DataAugmentor:
    """Handles data augmentation for training images."""

    def __init__(self, augmentation_factor: int = 9):
        """
        Initialize the augmentor.

        Args:
            augmentation_factor: Number of augmented images to generate per original.
        """
        self.augmentation_factor = augmentation_factor

    def augment(self, img: np.ndarray) -> List[tf.Tensor]:
        """
        Apply random augmentations to an image.

        Args:
            img: Input image as numpy array.

        Returns:
            List of augmented images.
        """
        augmented = []

        for _ in range(self.augmentation_factor):
            aug_img = tf.cast(img, tf.float32)

            # Random brightness
            aug_img = tf.image.stateless_random_brightness(
                aug_img, max_delta=0.02, seed=(1, 2)
            )

            # Random contrast
            aug_img = tf.image.stateless_random_contrast(
                aug_img, lower=0.6, upper=1, seed=(1, 3)
            )

            # Random horizontal flip
            aug_img = tf.image.stateless_random_flip_left_right(
                aug_img, seed=(np.random.randint(100), np.random.randint(100))
            )

            # Random JPEG quality
            aug_img = tf.image.stateless_random_jpeg_quality(
                aug_img,
                min_jpeg_quality=90,
                max_jpeg_quality=100,
                seed=(np.random.randint(100), np.random.randint(100)),
            )

            # Random saturation
            aug_img = tf.image.stateless_random_saturation(
                aug_img,
                lower=0.9,
                upper=1,
                seed=(np.random.randint(100), np.random.randint(100)),
            )

            augmented.append(aug_img)

        return augmented

    def augment_directory(self, directory: str) -> int:
        """
        Augment all images in a directory.

        Args:
            directory: Path to directory containing images.

        Returns:
            Total number of images after augmentation.
        """
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return 0

        original_files = os.listdir(directory)

        for file_name in original_files:
            img_path = os.path.join(directory, file_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue

            augmented_images = self.augment(img)

            for aug_img in augmented_images:
                new_path = os.path.join(directory, f"{uuid.uuid1()}.jpg")
                cv2.imwrite(new_path, aug_img.numpy())

        total_images = len(os.listdir(directory))
        print(f"Augmentation complete! Total images: {total_images}")
        return total_images


class DataLoader:
    """Handles loading and preparing datasets for training."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader.

        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(config)

    def load_dataset(
        self, max_samples: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load and prepare the training and test datasets.

        Args:
            max_samples: Maximum samples per class. If None, uses config default.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        max_samples = max_samples or self.config.MAX_SAMPLES

        # Load file paths
        anchor = tf.data.Dataset.list_files(
            os.path.join(self.config.paths.anc_path, "*.jpg")
        ).take(max_samples)

        positive = tf.data.Dataset.list_files(
            os.path.join(self.config.paths.pos_path, "*.jpg")
        ).take(max_samples)

        negative = tf.data.Dataset.list_files(
            os.path.join(self.config.paths.neg_path, "*.jpg")
        ).take(max_samples)

        # Create labeled datasets
        positives = tf.data.Dataset.zip(
            (
                anchor,
                positive,
                tf.data.Dataset.from_tensor_slices(tf.ones(len(list(anchor)))),
            )
        )

        # Re-create anchor dataset as it was consumed
        anchor = tf.data.Dataset.list_files(
            os.path.join(self.config.paths.anc_path, "*.jpg")
        ).take(max_samples)

        negatives = tf.data.Dataset.zip(
            (
                anchor,
                negative,
                tf.data.Dataset.from_tensor_slices(tf.zeros(max_samples)),
            )
        )

        # Concatenate datasets
        data = positives.concatenate(negatives)

        # Preprocess
        data = data.map(self.preprocessor.preprocess_twin)
        data = data.cache()
        data = data.shuffle(buffer_size=self.config.BUFFER_SIZE)

        # Split into train and test
        data_size = len(list(data))
        train_size = round(data_size * self.config.TRAIN_SPLIT)

        train_data = data.take(train_size)
        train_data = train_data.batch(self.config.BATCH_SIZE)
        train_data = train_data.prefetch(8)

        test_data = data.skip(train_size)
        test_data = test_data.batch(self.config.BATCH_SIZE)
        test_data = test_data.prefetch(8)

        print(f"Dataset loaded:")
        print(f"  Training samples: {train_size}")
        print(f"  Test samples: {data_size - train_size}")

        return train_data, test_data

    def get_dataset_info(self) -> dict:
        """
        Get information about the available data.

        Returns:
            Dictionary with counts for each data directory.
        """
        info = {}

        paths = {
            "anchor": self.config.paths.anc_path,
            "positive": self.config.paths.pos_path,
            "negative": self.config.paths.neg_path,
        }

        for name, path in paths.items():
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith(".jpg")]
                info[name] = len(files)
            else:
                info[name] = 0

        return info
