"""
Siamese Network Model for Face Verification.
"""

import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten

from .layers import L1Dist
from .config import Config


class SiameseModel:
    """
    Siamese Neural Network for face verification.

    This model learns to determine if two face images belong to the same person
    by computing embeddings and measuring their L1 distance.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Siamese Model.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()
        self.embedding: Optional[Model] = None
        self.siamese_model: Optional[Model] = None
        self._build_model()

    def _build_embedding(self) -> Model:
        """
        Build the embedding network.

        The embedding network converts face images into feature vectors.

        Returns:
            Keras Model that outputs embedding vectors.
        """
        inp = Input(shape=self.config.INPUT_SHAPE, name="input_image")

        # First convolutional block
        c1 = Conv2D(64, (10, 10), activation="relu")(inp)
        m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

        # Second convolutional block
        c2 = Conv2D(128, (7, 7), activation="relu")(m1)
        m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

        # Third convolutional block
        c3 = Conv2D(128, (4, 4), activation="relu")(m2)
        m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

        # Final embedding block
        c4 = Conv2D(256, (4, 4), activation="relu")(m3)
        f1 = Flatten()(c4)
        d1 = Dense(self.config.EMBEDDING_SIZE, activation="sigmoid")(f1)

        return Model(inputs=[inp], outputs=[d1], name="embedding")

    def _build_model(self) -> None:
        """Build the complete Siamese model."""
        # Build embedding network
        self.embedding = self._build_embedding()

        # Input layers
        input_image = Input(name="input_img", shape=self.config.INPUT_SHAPE)
        validation_image = Input(name="validation_img", shape=self.config.INPUT_SHAPE)

        # Get embeddings
        inp_embedding = self.embedding(input_image)
        val_embedding = self.embedding(validation_image)

        # Distance layer
        siamese_layer = L1Dist()
        siamese_layer._name = "distance"
        distances = siamese_layer(inp_embedding, val_embedding)

        # Classification layer
        classifier = Dense(1, activation="sigmoid")(distances)

        # Build the final model
        self.siamese_model = Model(
            inputs=[input_image, validation_image],
            outputs=classifier,
            name="SiameseNetwork",
        )

    def summary(self) -> None:
        """Print model summary."""
        if self.siamese_model:
            print("\n" + "=" * 50)
            print("Siamese Network Architecture")
            print("=" * 50)
            self.siamese_model.summary()

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            path: Path to save the model. If None, uses config default.

        Returns:
            Path where the model was saved.
        """
        save_path = path or self.config.paths.model_path
        self.siamese_model.save(save_path)
        print(f"Model saved to: {save_path}")
        return save_path

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a saved model from disk.

        Args:
            path: Path to load the model from. If None, uses config default.
        """
        load_path = path or self.config.paths.model_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at: {load_path}")

        self.siamese_model = tf.keras.models.load_model(
            load_path, custom_objects={"L1Dist": L1Dist}
        )
        print(f"Model loaded from: {load_path}")

    def predict(self, input_images: Tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Make predictions on image pairs.

        Args:
            input_images: Tuple of (anchor_images, validation_images).
            **kwargs: Additional keyword arguments for Keras predict (e.g., verbose).

        Returns:
            Prediction scores (0-1) for each pair.
        """
        return self.siamese_model.predict(input_images, **kwargs)

    def __call__(self, inputs, training: bool = False):
        """Allow the model to be called directly."""
        return self.siamese_model(inputs, training=training)

    @property
    def trainable_variables(self):
        """Return trainable variables of the model."""
        return self.siamese_model.trainable_variables
