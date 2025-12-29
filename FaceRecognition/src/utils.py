"""
Utility functions for Face Recognition project.
"""

import os
import shutil
from typing import Optional

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def setup_gpu() -> None:
    """
    Setup GPU memory growth to avoid OOM errors.
    Should be called at the start of the program.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU setup complete. Found {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Running on CPU.")


def print_tensorflow_info() -> None:
    """Print TensorFlow version and device information."""
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    cpus = tf.config.experimental.list_physical_devices("CPU")

    print(f"Available GPUs: {len(gpus)}")
    print(f"Available CPUs: {len(cpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")


def visualize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    label: Optional[float] = None,
    prediction: Optional[float] = None,
) -> None:
    """
    Visualize a pair of images side by side.

    Args:
        img1: First image.
        img2: Second image.
        label: Ground truth label (optional).
        prediction: Model prediction (optional).
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Validation Image")
    plt.axis("off")

    title = ""
    if label is not None:
        title += f"Label: {'Same' if label > 0.5 else 'Different'}"
    if prediction is not None:
        if title:
            title += " | "
        title += f"Prediction: {prediction:.2%}"

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary with 'loss', 'recall', and 'precision' keys.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Recall
    axes[1].plot(history["recall"])
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].grid(True)

    # Precision
    axes[2].plot(history["precision"])
    axes[2].set_title("Precision")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Precision")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def export_model_for_download(
    model_path: str, checkpoint_dir: str, output_dir: str
) -> dict:
    """
    Create zip archives of model and checkpoints for easy download.
    Useful for Kaggle environment.

    Args:
        model_path: Path to the .keras model file.
        checkpoint_dir: Path to the checkpoints directory.
        output_dir: Directory to save the zip files.

    Returns:
        Dictionary with paths to the created zip files.
    """
    os.makedirs(output_dir, exist_ok=True)

    result = {}

    # Export model
    if os.path.exists(model_path):
        model_zip = os.path.join(output_dir, "model_backup")
        shutil.make_archive(
            model_zip, "zip", os.path.dirname(model_path), os.path.basename(model_path)
        )
        result["model"] = model_zip + ".zip"
        print(f"Model exported: {result['model']}")

    # Export checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoint_zip = os.path.join(output_dir, "checkpoints_backup")
        shutil.make_archive(checkpoint_zip, "zip", checkpoint_dir)
        result["checkpoints"] = checkpoint_zip + ".zip"
        print(f"Checkpoints exported: {result['checkpoints']}")

    return result


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    List all available checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        List of checkpoint paths.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    checkpoint = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)

    checkpoints = manager.checkpoints

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    else:
        print("No checkpoints found.")

    return list(checkpoints)
