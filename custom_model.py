##########
#
# Custom ConvNet for Minecraft Pig Chase :
# This model is ConvNet for custom shape (120, 160, 3).
# The output shape is (256, ) which is used in AttentionNet.
#
##########
from typing import Dict, List
import gym
import numpy as np
import tensorflow as tf

from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.tf.misc import normc_initializer

OBS_STACK_SIZE = 3      # channel size for observations

class MyCustomVisionNetwork(TFModelV2):
    def __init__(self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str):

        super(MyCustomVisionNetwork, self).__init__(obs_space, action_space,
            num_outputs, model_config, name)

        ##########
        # Prepare inputs
        ##########

        input_obs = tf.keras.layers.Input(
            shape=(120, 160, OBS_STACK_SIZE),
            name="observations")

        ##########
        # Build action model
        # This is the input for attention network
        ##########

        # ConvNet :
        # (?, 120, 160, channel) -> (?, 40, 40, 16)
        last_layer = tf.keras.layers.Conv2D(
            16,
            [12, 16],
            strides=(3, 4),
            activation="relu",
            padding="same",
            data_format="channels_last",
            kernel_initializer=normc_initializer(0.01),
            bias_initializer=normc_initializer(0.01),
            name="conv_action01")(input_obs)
        # ConvNet :
        # -> (?, 10, 10, 32)
        last_layer = tf.keras.layers.Conv2D(
            32,
            [6, 6],
            strides=(4, 4),
            activation="relu",
            padding="same",
            data_format="channels_last",
            kernel_initializer=normc_initializer(0.01),
            bias_initializer=normc_initializer(0.01),
            name="conv_action02")(last_layer)
        # ConvNet :
        # -> (?, 1, 1, 256)
        last_layer = tf.keras.layers.Conv2D(
            256,
            [10, 10],
            strides=(1, 1),
            activation="relu",
            padding="valid",
            data_format="channels_last",
            kernel_initializer=normc_initializer(0.01),
            bias_initializer=normc_initializer(0.01),
            name="conv_action03")(last_layer)

        # (?, 1, 1, 256) -> (?, 256)
        #conv_layer = tf.keras.layers.Lambda(
        #    lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)
        out_action = tf.keras.layers.Flatten(
            data_format="channels_last")(last_layer)

        # Set self.num_outputs for attention net
        self.num_outputs = 256
        self.data_format = "channels_last"
        self.last_layer_is_flattened = True

        ##########
        # Build value model
        # (Separated model, even when vf_share_layers=True.)
        ##########

        # ConvNet :
        # (?, 120, 160, channel) -> (?, 40, 40, 16)
        last_layer = tf.keras.layers.Conv2D(
            16,
            [12, 16],
            strides=(3, 4),
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_value01")(input_obs)
        # ConvNet :
        # -> (?, 10, 10, 32)
        last_layer = tf.keras.layers.Conv2D(
            32,
            [6, 6],
            strides=(4, 4),
            activation="relu",
            padding="same",
            data_format="channels_last",
            name="conv_value02")(last_layer)
        # ConvNet :
        # -> (?, 1, 1, 256)
        last_layer = tf.keras.layers.Conv2D(
            256,
            [10, 10],
            strides=(1, 1),
            activation="relu",
            padding="valid",
            data_format="channels_last",
            name="conv_value03")(last_layer)
        # ConvNet :
        # -> (?, 1, 1, 1)
        last_layer = tf.keras.layers.Conv2D(
            1, [1, 1],
            activation=None,
            padding="same",
            data_format="channels_last",
            name="conv_value04")(last_layer)

        # (?, 1, 1, 1) -> (?, 1)
        out_value = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)

        ##########
        # Set base model
        ##########

        self.base_model = tf.keras.Model(
            [input_obs],
            [out_action, out_value])
        # No longer needed in version 1.4
        # self.register_variables(self.base_model.variables)

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        # Note : Input Data Structure
        # input_dict["obs_flat"] is :
        #   Tensor("default_policy/flatten/Reshape:0", shape=(?, 57600), dtype=float32)
        # input_dict["obs"] is :
        #   Tensor("default_policy/obs:0", shape=(?, 120, 160, 3), dtype=float32)

        # Pass to model
        model_out, self._value_out = self.base_model([input_dict["obs"]])

        return model_out, state

    # This is needed on "critic" process
    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
