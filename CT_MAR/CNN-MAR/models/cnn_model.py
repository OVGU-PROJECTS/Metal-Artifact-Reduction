# cnn_model.py
"""
Zhang & Yu CNN-MAR network — TensorFlow/Keras implementation.

Architecture:
- 5 convolutional layers
- kernel size 3x3 in each layer
- 32 feature maps in layers 1-4, 1 output channel in layer 5
- padding=1 ("same") so output size equals input size
- ReLU after layers 1-4, no activation in the last layer

Training loss:
- Loss is mean squared error (Frobenius norm squared averaged over samples)

Source: Zhang & Yu, IEEE TMI 2018.
"""
from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, initializers


@tf.keras.utils.register_keras_serializable(package="cnnmar")
class ZhangCNNMAR(tf.keras.Model):
    def __init__(self, *, nMC: int = 3, nFeature: int = 32, nKernel: int = 3, name: str = "ZhangCNNMAR", **kwargs):
        super().__init__(name=name, **kwargs)
        if nKernel != 3:
            # The paper uses 3×3 kernels throughout. citeturn7view2
            raise ValueError("Paper uses nKernel=3 (3x3). Keep nKernel=3 for faithful replication.")
        if nMC != 3:
            # The paper uses 3-channel input (uncorrected, BHC, LI). citeturn8view0
            raise ValueError("Paper uses nMC=3 input channels (raw, BHC, LI). Keep nMC=3 for faithful replication.")
        if nFeature != 32:
            # The paper uses 32 feature maps in layers 1–4. citeturn7view2
            raise ValueError("Paper uses nFeature=32. Keep nFeature=32 for faithful replication.")

        self.nMC = int(nMC)
        self.nFeature = int(nFeature)
        self.nKernel = int(nKernel)

        # MatConvNet examples commonly use Xavier/Glorot init; the paper does not mandate an init.
        kinit = initializers.GlorotUniform()
        binit = "zeros"

        self.conv1 = layers.Conv2D(self.nFeature, self.nKernel, padding="same",
                                   use_bias=True, kernel_initializer=kinit, bias_initializer=binit, name="conv1")
        self.relu1 = layers.ReLU(name="relu1")

        self.conv2 = layers.Conv2D(self.nFeature, self.nKernel, padding="same",
                                   use_bias=True, kernel_initializer=kinit, bias_initializer=binit, name="conv2")
        self.relu2 = layers.ReLU(name="relu2")

        self.conv3 = layers.Conv2D(self.nFeature, self.nKernel, padding="same",
                                   use_bias=True, kernel_initializer=kinit, bias_initializer=binit, name="conv3")
        self.relu3 = layers.ReLU(name="relu3")

        self.conv4 = layers.Conv2D(self.nFeature, self.nKernel, padding="same",
                                   use_bias=True, kernel_initializer=kinit, bias_initializer=binit, name="conv4")
        self.relu4 = layers.ReLU(name="relu4")

        self.pred = layers.Conv2D(1, self.nKernel, padding="same",
                                  use_bias=True, kernel_initializer=kinit, bias_initializer=binit, name="prediction")

    def call(self, x, training=False):
        x = self.relu1(self.conv1(x, training=training))
        x = self.relu2(self.conv2(x, training=training))
        x = self.relu3(self.conv3(x, training=training))
        x = self.relu4(self.conv4(x, training=training))
        x = self.pred(x, training=training)
        return x

    def get_config(self):
        return {"nMC": self.nMC, "nFeature": self.nFeature, "nKernel": self.nKernel, "name": self.name}

    @staticmethod
    @tf.keras.utils.register_keras_serializable(package="cnnmar")
    def s_similarity(y_true, y_pred, hu_min=-1000.0, hu_max=2000.0, mu_water=0.02):
        # Training tensors are in attenuation units; SSIM is reported in HU.
        y_true = y_true[..., :1]
        y_pred = y_pred[..., :1]
        y_true = y_true * (1000.0 / mu_water) - 1000.0
        y_pred = y_pred * (1000.0 / mu_water) - 1000.0

        # clip then normalize to [0,1]
        y_true = tf.clip_by_value(y_true, hu_min, hu_max)
        y_pred = tf.clip_by_value(y_pred, hu_min, hu_max)
        denom = (hu_max - hu_min)
        y_true01 = (y_true - hu_min) / denom
        y_pred01 = (y_pred - hu_min) / denom

        # returns [B] then Keras averages over batch
        return tf.image.ssim(y_true01, y_pred01, max_val=1.0)
