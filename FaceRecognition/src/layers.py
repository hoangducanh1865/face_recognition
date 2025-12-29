"""
Custom layers for the Siamese Network.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class L1Dist(Layer):
    """
    Custom L1 Distance Layer for Siamese Network.

    Computes the absolute difference between two embeddings,
    which serves as the distance metric for face verification.

    This layer is compatible with TensorFlow 2.16+ / Keras 3.
    """

    def __init__(self, **kwargs):
        """Initialize the L1 Distance layer."""
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        """
        Compute L1 distance between two embeddings.

        Args:
            input_embedding: Embedding from the input (anchor) image.
            validation_embedding: Embedding from the validation image.

        Returns:
            Absolute difference between the two embeddings.
        """
        # Handle case where inputs might be wrapped in lists (Keras 3 compatibility)
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]

        return tf.math.abs(input_embedding - validation_embedding)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        return config
