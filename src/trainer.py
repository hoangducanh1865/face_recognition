"""
Training module for the Siamese Network.
"""

import os
from typing import Optional, Callable

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

from .config import Config
from .model import SiameseModel


class Trainer:
    """Handles training of the Siamese Network."""

    def __init__(self, model: SiameseModel, config: Optional[Config] = None):
        """
        Initialize the trainer.

        Args:
            model: The Siamese model to train.
            config: Configuration object.
        """
        self.model = model
        self.config = config or Config()

        # Loss and optimizer
        self.loss_fn = tf.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(self.config.LEARNING_RATE)

        # Checkpoint setup
        self._setup_checkpoint()

    def _setup_checkpoint(self) -> None:
        """Setup checkpoint manager for saving training progress."""
        os.makedirs(self.config.paths.checkpoint_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            opt=self.optimizer, siamese_model=self.model.siamese_model
        )

        self.checkpoint_prefix = os.path.join(self.config.paths.checkpoint_dir, "ckpt")

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.config.paths.checkpoint_dir, max_to_keep=5
        )

    def restore_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Restore from a checkpoint.

        Args:
            checkpoint_path: Specific checkpoint path. If None, restores latest.

        Returns:
            True if checkpoint was restored, False otherwise.
        """
        if checkpoint_path:
            status = self.checkpoint.restore(checkpoint_path)
            print(f"Restored from checkpoint: {checkpoint_path}")
            return True

        latest = self.checkpoint_manager.latest_checkpoint
        if latest:
            status = self.checkpoint.restore(latest)
            print(f"Restored from latest checkpoint: {latest}")
            return True

        print("No checkpoint found. Starting fresh training.")
        return False

    @tf.function
    def _train_step(self, batch) -> tf.Tensor:
        """
        Execute a single training step.

        Args:
            batch: A batch of training data.

        Returns:
            Loss value for this batch.
        """
        with tf.GradientTape() as tape:
            # Get anchor and validation images
            X = batch[:2]
            # Get labels
            y = batch[2]

            # Forward pass
            yhat = self.model(X, training=True)
            # Calculate loss
            loss = self.loss_fn(y, yhat)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(
        self,
        train_data: tf.data.Dataset,
        epochs: Optional[int] = None,
        checkpoint_frequency: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_data: Training dataset.
            epochs: Number of epochs to train.
            checkpoint_frequency: Save checkpoint every N epochs.
            callback: Optional callback function called after each epoch.

        Returns:
            Dictionary containing training history.
        """
        epochs = epochs or self.config.DEFAULT_EPOCHS
        checkpoint_frequency = checkpoint_frequency or self.config.CHECKPOINT_FREQUENCY

        history = {"loss": [], "recall": [], "precision": []}

        print(f"\n{'='*50}")
        print(f"Starting Training")
        print(f"{'='*50}")
        print(f"Epochs: {epochs}")
        print(f"Checkpoint Frequency: Every {checkpoint_frequency} epochs")
        print(f"{'='*50}\n")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Progress bar
            progbar = tf.keras.utils.Progbar(len(train_data))

            # Metrics
            recall = Recall()
            precision = Precision()
            epoch_loss = 0.0

            # Training loop
            for idx, batch in enumerate(train_data):
                loss = self._train_step(batch)
                epoch_loss = loss.numpy()

                # Update metrics
                yhat = self.model.predict(batch[:2])
                recall.update_state(batch[2], yhat)
                precision.update_state(batch[2], yhat)

                progbar.update(idx + 1)

            # Record metrics
            epoch_recall = recall.result().numpy()
            epoch_precision = precision.result().numpy()

            history["loss"].append(epoch_loss)
            history["recall"].append(epoch_recall)
            history["precision"].append(epoch_precision)

            print(
                f"Loss: {epoch_loss:.4f}, Recall: {epoch_recall:.4f}, "
                f"Precision: {epoch_precision:.4f}"
            )

            # Save checkpoint
            if epoch % checkpoint_frequency == 0:
                save_path = self.checkpoint_manager.save()
                print(f"Checkpoint saved: {save_path}")

            # Call callback if provided
            if callback:
                callback(epoch, history)

        print(f"\n{'='*50}")
        print("Training Complete!")
        print(f"{'='*50}\n")

        return history

    def evaluate(self, test_data: tf.data.Dataset) -> dict:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test dataset.

        Returns:
            Dictionary containing evaluation metrics.
        """
        recall = Recall()
        precision = Precision()

        print("\nEvaluating model...")

        for test_input, test_val, y_true in test_data.as_numpy_iterator():
            yhat = self.model.predict([test_input, test_val])
            recall.update_state(y_true, yhat)
            precision.update_state(y_true, yhat)

        results = {
            "recall": recall.result().numpy(),
            "precision": precision.result().numpy(),
        }

        print(f"\nEvaluation Results:")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")

        return results
