import tensorflow as tf

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution, DiagGaussian
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, List

class MyCustomDiagGaussian(DiagGaussian):
    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        # Control feature vector size of model output.
        # In this case, 2 values for mean. (log_std is a fixed value.)
        return 2

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2):
        # Add bias to start with balanced inputs
        inputs_bias = tf.math.add(inputs, [-1.0])
        # Get mean by the difference between normalized left (y1) and right (y2) :
        #     y1  = x1 / (x1^2 + x2^2)^(1/2) and
        #     y2 = x2 / (x1^2 + x2^2)^(1/2)
        # where x1 is origin left and x2 is origin right
        inputs_norm = tf.math.l2_normalize(inputs_bias, axis=1)
        mean_left, mean_right = tf.split(inputs_norm, [1, 1], axis=1)
        mean_squash = tf.math.subtract(mean_right, mean_left)
        mean_squash = tf.math.divide(mean_squash, [1.42]) # because absolute max value is sqrt(2)
        # log_std is a constant, -2.3
        batch_size = tf.shape(mean_squash)[0]
        log_std = tf.fill([batch_size, 1], -2.3)
        # Initialize DiagGaussian
        inputs_new = tf.concat([mean_squash, log_std], axis=1)
        super().__init__(inputs_new, model)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        sample_op = self.mean + self.std * tf.random.normal(tf.shape(self.mean))
        return tf.clip_by_value(sample_op, -1.0, 1.0)
