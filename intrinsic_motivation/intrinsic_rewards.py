# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for generative models used to derive intrinsic rewards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from dopamine.discrete_domains import atari_lib
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

import gin
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.summary import SummaryWriter as TFSummaryWriter
from tensorflow.compat.v1.summary import FileWriter as TFFileWriter
from flax.metrics.tensorboard import SummaryWriter as FlaxSummaryWriter
# from tensorflow.contrib import slim
import tf_slim as slim
import cv2
import zlib


import time

PSEUDO_COUNT_QUANTIZATION_FACTOR = 8
PSEUDO_COUNT_OBSERVATION_SHAPE = (42, 42)
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
TRAIN_START_AMOUNT = 10000

class Timer:
    def __init__(self, name="") -> None:
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
          print(f"{self.name} took {self.interval}")

def timing_wrapper(func):
  def wrap_it(*args, **kwargs):
    with Timer() as t:
      result = func(*args, **kwargs)
    print('{} took {} seconds'.format(func.__name__, t.interval))
    return result
  return wrap_it

@slim.add_arg_scope
def masked_conv2d(inputs, num_outputs, kernel_size,
                  activation_fn=tf.nn.relu,
                  weights_initializer=tf.initializers.glorot_normal(),
                  biases_initializer=tf.initializers.zeros(),
                  stride=(1, 1),
                  scope=None,
                  mask_type='A',
                  collection=None,
                  output_multiplier=1):
  """Creates masked convolutions used in PixelCNN.

  There are two types of masked convolutions, type A and B, see Figure 1 in
  https://arxiv.org/abs/1606.05328 for more details.

  Args:
    inputs: input image.
    num_outputs: int, number of filters used in the convolution.
    kernel_size: int, size of convolution kernel.
    activation_fn: activation function used after the convolution.
    weights_initializer: distribution used to initialize the kernel.
    biases_initializer: distribution used to initialize biases.
    stride: convolution stride.
    scope: name of the tensorflow scope.
    mask_type: type of masked convolution, must be A or B.
    collection: tf variables collection.
    output_multiplier: number of convolutional network stacks.

  Returns:
    frame post convolution.
  """
  assert mask_type in ('A', 'B') and num_outputs % output_multiplier == 0
  num_inputs = int(inputs.get_shape()[-1])
  kernel_shape = tuple(kernel_size) + (num_inputs, num_outputs)
  strides = (1,) + tuple(stride) + (1,)
  biases_shape = [num_outputs]

  mask_list = [np.zeros(
      tuple(kernel_size) + (num_inputs, num_outputs // output_multiplier),
      dtype=np.float32) for _ in range(output_multiplier)]
  for i in range(output_multiplier):
    # Mask type A
    if kernel_shape[0] > 1:
      mask_list[i][:kernel_shape[0]//2] = 1.0
    if kernel_shape[1] > 1:
      mask_list[i][kernel_shape[0]//2, :kernel_shape[1]//2] = 1.0
    # Mask type B
    if mask_type == 'B':
      mask_list[i][kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
  mask_values = np.concatenate(mask_list, axis=3)

  with tf.variable_scope(scope):
    w = tf.get_variable('W', kernel_shape, trainable=True,
                        initializer=weights_initializer)
    b = tf.get_variable('biases', biases_shape, trainable=True,
                        initializer=biases_initializer)
    if collection is not None:
      tf.add_to_collection(collection, w)
      tf.add_to_collection(collection, b)

    mask = tf.constant(mask_values, dtype=tf.float32)
    mask.set_shape(kernel_shape)

    convolution = tf.nn.conv2d(inputs, mask * w, strides, padding='SAME')
    convolution_bias = tf.nn.bias_add(convolution, b)

    if activation_fn is not None:
      convolution_bias = activation_fn(convolution_bias)
  return convolution_bias


def gating_layer(x, embedding, hidden_units, scope_name=''):
  """Create the gating layer used in the PixelCNN architecture."""
  with tf.variable_scope(scope_name):
    out = masked_conv2d(x, 2*hidden_units, [3, 3],
                        mask_type='B',
                        activation_fn=None,
                        output_multiplier=2,
                        scope='masked_conv')
    out += slim.conv2d(embedding, 2*hidden_units, [1, 1],
                       activation_fn=None)
    out = tf.reshape(out, [-1, 2])
    out = tf.tanh(out[:, 0]) + tf.sigmoid(out[:, 1])
  return tf.reshape(out, x.get_shape())


@gin.configurable
class CTSIntrinsicReward(object):
  """Class used to instantiate a CTS density model used for exploration."""

  def __init__(self,
               reward_scale,
               convolutional=False,
               observation_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,
               quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR):
    """Constructor.

    Args:
      reward_scale: float, scale factor applied to the raw rewards.
      convolutional: bool, whether to use convolutional CTS.
      observation_shape: tuple, 2D dimensions of the observation predicted
        by the model. Needs to be square.
      quantization_factor: int, number of bits for the predicted image
    Raises:
      ValueError: when the `observation_shape` is not square.
    """
    self._reward_scale = reward_scale
    if  (len(observation_shape) != 2
         or observation_shape[0] != observation_shape[1]):
      raise ValueError('Observation shape needs to be square')
    self._observation_shape = observation_shape
    self.density_model = shannon.CTSTensorModel(
        observation_shape, convolutional)
    self._quantization_factor = quantization_factor

  def update(self, observation):
    """Updates the density model with the given observation.

    Args:
      observation: Input frame.

    Returns:
      Update log-probability.
    """
    input_tensor = self._preprocess(observation)
    return self.density_model.Update(input_tensor)

  def compute_intrinsic_reward(self, observation, training_steps, eval_mode, action=None):
    """Updates the model, returns the intrinsic reward.

    Args:
      observation: Input frame. For compatibility with other models, this
        may have a batch-size of 1 as its first dimension.
      training_steps: int, number of training steps.
      eval_mode: bool, whether or not running eval mode.

    Returns:
      The corresponding intrinsic reward.
    """
    del training_steps
    input_tensor = self._preprocess(observation)
    if not eval_mode:
      log_rho_t = self.density_model.Update(input_tensor)
      log_rho_tp1 = self.density_model.LogProb(input_tensor)
      ipd = log_rho_tp1 - log_rho_t
    else:
      # Do not update the density model in evaluation mode
      ipd = self.density_model.IPD(input_tensor)

    # Compute the pseudo count
    ipd_clipped = min(ipd, 25)
    inv_pseudo_count = max(0, math.expm1(ipd_clipped))
    reward = float(self._reward_scale) * math.sqrt(inv_pseudo_count)
    return reward

  def _preprocess(self, observation):
    """Converts the given observation into something the model can use.

    Args:
      observation: Input frame.

    Returns:
      Processed frame.

    Raises:
      ValueError: If observation provided is not 2D.
    """
    if observation.ndim != 2:
      raise ValueError('Observation needs to be 2D.')
    input_tensor = cv2.resize(observation,
                              self._observation_shape,
                              interpolation=cv2.INTER_AREA)
    input_tensor //= (256 // self._quantization_factor)
    # Convert to signed int (this may be unpleasantly inefficient).
    input_tensor = input_tensor.astype('i', copy=False)
    return input_tensor


@gin.configurable
class PixelCNNIntrinsicReward(object):
  """PixelCNN class to instantiate a bonus using a PixelCNN density model."""

  def __init__(self,
               sess,
               reward_scale,
               ipd_scale,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               resize_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,
               quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR,
               tf_device='/cpu:*',
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.0001,
                   momentum=0.9,
                   epsilon=0.0001)):
    self._sess = sess
    self.reward_scale = reward_scale
    self.ipd_scale = ipd_scale
    self.observation_shape = observation_shape
    self.resize_shape = resize_shape
    self.quantization_factor = quantization_factor
    self.optimizer = optimizer

    with tf.device(tf_device), tf.name_scope('intrinsic_pixelcnn'):
      observation_shape = (1,) + observation_shape + (1,)
      self.obs_ph = tf.placeholder(tf.uint8, shape=observation_shape,
                                   name='obs_ph')
      self.preproccessed_obs = self._preprocess(self.obs_ph, resize_shape)
      self.iter_ph = tf.placeholder(tf.uint32, shape=[], name='iter_num')
      self.eval_ph = tf.placeholder(tf.bool, shape=[], name='eval_mode')
      self.network = tf.make_template('PixelCNN', self._network_template)
      self.ipd = tf.cond(tf.logical_not(self.eval_ph),
                         self.update,
                         self.virtual_update)
      self.reward = self.ipd_to_reward(self.ipd, self.iter_ph)

  # @timing_wrapper
  def compute_intrinsic_reward(self, observation, training_steps, eval_mode, action=None):
    """Updates the model (during training), returns the intrinsic reward.

    Args:
      observation: Input frame. For compatibility with other models, this
        may have a batch-size of 1 as its first dimension.
      training_steps: Number of training steps, int.
      eval_mode: bool, whether or not running eval mode.

    Returns:
      The corresponding intrinsic reward.
    """
    observation = observation[np.newaxis, :, :, np.newaxis]
    return self._sess.run(self.reward, {self.obs_ph: observation,
                                        self.iter_ph: training_steps,
                                        self.eval_ph: eval_mode})

  def _preprocess(self, obs, obs_shape):
    """Preprocess the input."""
    obs = tf.cast(obs, tf.float32)
    obs = tf.image.resize_bilinear(obs, obs_shape)
    denom = tf.constant(256 // self.quantization_factor, dtype=tf.float32)
    return tf.floordiv(obs, denom)

  @gin.configurable
  def _network_template(self, obs, num_layers, hidden_units):
    """PixelCNN network architecture."""
    with slim.arg_scope([slim.conv2d, masked_conv2d],
                        weights_initializer=tf.variance_scaling_initializer(
                            distribution='uniform'),
                        biases_initializer=tf.constant_initializer(0.0)):
      net = masked_conv2d(obs, hidden_units, [7, 7], mask_type='A',
                          activation_fn=None, scope='masked_conv_1')

      embedding = slim.model_variable(
          'embedding',
          shape=(1,) + self.resize_shape + (4,),
          initializer=tf.variance_scaling_initializer(
              distribution='uniform'))
      for i in range(1, num_layers + 1):
        net2 = gating_layer(net, embedding, hidden_units,
                            'gating_{}'.format(i))
        net += masked_conv2d(net2, hidden_units, [1, 1],
                             mask_type='B',
                             activation_fn=None,
                             scope='masked_conv_{}'.format(i+1))

      net += slim.conv2d(embedding, hidden_units, [1, 1],
                         activation_fn=None)
      net = tf.nn.relu(net)
      net = masked_conv2d(net, 64, [1, 1], scope='1x1_conv_out',
                          mask_type='B',
                          activation_fn=tf.nn.relu)
      logits = masked_conv2d(net, self.quantization_factor, [1, 1],
                             scope='logits', mask_type='B',
                             activation_fn=None)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.cast(obs, tf.int32),
        logits=logits,
        reduction=tf.losses.Reduction.MEAN)
    return collections.namedtuple('PixelCNN_network', ['logits', 'loss'])(
        logits, loss)

  def update(self):
    """Computes the log likehood difference and update the density model."""
    with tf.name_scope('update'):
      with tf.name_scope('pre_update'):
        loss = self.network(self.preproccessed_obs).loss

      train_op = self.optimizer.minimize(loss)

      with tf.name_scope('post_update'), tf.control_dependencies([train_op]):
        loss_post_training = self.network(self.preproccessed_obs).loss
        ipd = (loss - loss_post_training) * (
            self.resize_shape[0] * self.resize_shape[1])
    return ipd

  def virtual_update(self):
    """Computes the log likelihood difference without updating the network."""
    with tf.name_scope('virtual_update'):
      with tf.name_scope('pre_update'):
        loss = self.network(self.preproccessed_obs).loss

      grads_and_vars = self.optimizer.compute_gradients(loss)
      model_vars = [gv[1] for gv in grads_and_vars]
      saved_vars = [tf.Variable(v.initialized_value()) for v in model_vars]
      backup_op = tf.group(*[t.assign(s)
                             for t, s in zip(saved_vars, model_vars)])
      with tf.control_dependencies([backup_op]):
        train_op = self.optimizer.apply_gradients(grads_and_vars)
      with tf.control_dependencies([train_op]), tf.name_scope('post_update'):
        loss_post_training = self.network(self.preproccessed_obs).loss
      with tf.control_dependencies([loss_post_training]):
        restore_op = tf.group(*[d.assign(s)
                                for d, s in zip(model_vars, saved_vars)])
      with tf.control_dependencies([restore_op]):
        ipd = (loss - loss_post_training) * \
              self.resize_shape[0] * self.resize_shape[1]
      return ipd

  def ipd_to_reward(self, ipd, steps):
    """Computes the intrinsic reward from IPD."""
    # Prediction gain decay
    ipd = self.ipd_scale * ipd / tf.sqrt(tf.to_float(steps))
    inv_pseudo_count = tf.maximum(tf.expm1(ipd), 0.0)
    return self.reward_scale * tf.sqrt(inv_pseudo_count)


class CoinFlipMaker(object):
  def __init__(self, output_dimensions, p_replace, only_zero_flips=False):
    self.p_replace = p_replace
    self.output_dimensions = output_dimensions
    self.only_zero_flips = only_zero_flips
    self.previous_output = self._draw()

  def _draw(self):
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    return 2 * np.random.binomial(1, 0.5, size=self.output_dimensions) - 1

  def __call__(self):
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    new_output = self._draw()
    new_output = np.where(
      np.random.rand(self.output_dimensions) < self.p_replace,
      new_output,
      self.previous_output
    )
    self.previous_output = new_output
    return new_output

  def reset(self):
    if self.only_zero_flips:
      self.previous_output = np.zeros(self.output_dimensions, dtype=np.float32)
    self.previous_output = self._draw()

class NumUpdatesBuffer:
  def __init__(self, max_size):
    self.max_size = max_size
    self.num_updates = np.zeros((max_size, ), dtype=np.float32)
    self.ptr = 0
    self.size = 0

  def add(self, num_updates=0.):
    self.num_updates[self.ptr] = num_updates
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def increment_priorities(self, indices):
    assert len(indices.shape) == 1, indices.shape
    self.num_updates[indices] = self.num_updates[indices] + 1

  def sample(self, indices):
    assert len(indices.shape) == 1, indices.shape
    return self.num_updates[indices]


def _print_first_nonzero(arr):
  if arr is None:
    print("arr is None")
    return None
  for i in range(arr.shape[0]):
    for j in range(arr[i].shape[0]):
      if arr[i][j] != 0:
        print(f"({i}, {j}): {arr[i][j]}")
        return (i ,j)
  raise Exception('huh')

def get_direction(start, end):
  if start is None or end is None:
    return None
  mapping = {
    (0, -1) : 0,
    (0, 1) : 1,
    (-1, 0) : 2,
    (1, 0): 3,
    (0, 0): "Wall"
  }
  return mapping[tuple(np.array(end) - np.array(start))]



@gin.configurable
class CoinFlipCounterIntrinsicReward(object):
  """Our thang"""

  def __init__(self,
               sess,
               reward_scale,
               ipd_scale,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               resize_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,
               quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR,
               intrinsic_replay_start_size=TRAIN_START_AMOUNT,
               intrinsic_replay_reward_add_start_size=TRAIN_START_AMOUNT,
               intrinsic_replay_buffer_size=10**6,
               tf_device='/cpu:*',
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.0001,
                   momentum=0.9,
                   epsilon=0.0001),
               output_dimensions=100, # This is going to be in the config?
               batch_size=32,
               prioritization_strategy="combination",
               use_prioritized_buffer=True,
               priority_alpha=0.5,
               use_final_tanh=False,
               update_period=1,
               p_replace=1.,
               summary_writer=None,
               use_random_prior=True,
               use_lwm_representation_learning=False,
               lwm_representation_learning_scale=1.0,
               use_icm_representation_learning=False,
               icm_representation_learning_scale=1.0,
               use_representation_whitening=True,
               use_count_consistency=False,
               count_consistency_scale=1.0,
               use_reward_normalization=False,
               shared_representation_learning_latent=False,
               bonus_exponent=0.5,
               share_dqn_conv=False,
               agent=None,
               use_fresh_rewards=False,
               num_actions=None,
               only_zero_flips=False,
               continuous_control=False,
               use_observation_normalization=False,
               n_action_dims=None,
               ):
    # We now have 3 competing prioritization types.
    assert not (continuous_control and use_icm_representation_learning), "can't do cc and icm at this time"
    assert not (continuous_control and share_dqn_conv), "can't do cc and share dqn conv at this time"
    assert prioritization_strategy in ("exponential_average", "equalizing", "combination"), prioritization_strategy
    assert priority_alpha >= 0. and priority_alpha <= 1.
    assert isinstance(n_action_dims, int) or not continuous_control, type(n_action_dims)

    num_repr_enabled = sum(map(int, [use_lwm_representation_learning, use_icm_representation_learning, use_count_consistency]))
    assert num_repr_enabled <= 1, "Only one representation learning method can be enabled at a time."

    if use_fresh_rewards:
      assert not share_dqn_conv, "Can't use fresh rewards and share dqn conv"

    assert intrinsic_replay_reward_add_start_size >= intrinsic_replay_start_size

    self._sess = sess
    self.output_dimensions = output_dimensions
    self.reward_scale = reward_scale
    self.ipd_scale = ipd_scale # we don't actually use this, but don't want to remove yet.
    self.observation_shape = observation_shape
    self.continuous_control = continuous_control

    if isinstance(resize_shape, int):
      resize_shape = (resize_shape, resize_shape)
    self.resize_shape = resize_shape
    
    self.n_action_dims = n_action_dims

    self.quantization_factor = quantization_factor
    self.optimizer = optimizer
    self.intrinsic_replay_start_size = intrinsic_replay_start_size
    self.intrinsic_replay_reward_add_start_size = intrinsic_replay_reward_add_start_size
    self.intrinsic_replay_buffer_size = intrinsic_replay_buffer_size
    self.batch_size = batch_size
    self.use_prioritized_buffer = use_prioritized_buffer
    self.prioritization_strategy = prioritization_strategy
    self.priority_alpha = priority_alpha
    self.use_final_tanh = use_final_tanh
    self.update_period = update_period
    self.summary_writer = summary_writer
    if use_random_prior and use_final_tanh:
      raise Exception("These don't mesh")
    self.use_random_prior = use_random_prior
    self.use_lwm_representation_learning = use_lwm_representation_learning
    self.lwm_representation_learning_scale = lwm_representation_learning_scale
    self.use_representation_whitening = use_representation_whitening
    self.use_count_consistency = use_count_consistency
    self.count_consistency_scale = count_consistency_scale
    self.use_icm_representation_learning = use_icm_representation_learning
    self.icm_representation_learning_scale = icm_representation_learning_scale
    # If true, LWM acts directly on the coinflips!
    self.shared_representation_learning_latent = shared_representation_learning_latent
    self.use_reward_normalization = use_reward_normalization
    self.bonus_exponent = bonus_exponent
    self.share_dqn_conv = share_dqn_conv
    if share_dqn_conv:
      assert tuple(observation_shape) == tuple(resize_shape), "If sharing conv layer, cannot resize"
    self._agent = agent # for sharing the conv... not loving it.
    self.use_fresh_rewards = use_fresh_rewards
    self.num_actions = num_actions
    self.only_zero_flips = only_zero_flips
    self.use_observation_normalization = use_observation_normalization
    self._t = 0

    # when this gets too big, we run out of memory. Shucks.
    # Seems unavoidable, 44*44*8 = 15,000. * 4 (float32) is 60 gb RAM just for that. Jeez. 


    self.previous_obs = None
    self.last_action = None
    self.replay_buffer = self.cc_make_cfn_replay_buffer() if self.continuous_control else self.make_cfn_replay_buffer()
    if self.use_prioritized_buffer:
      self.num_updates_buffer = NumUpdatesBuffer(max_size=self.intrinsic_replay_buffer_size)


    self.coin_flip_maker = CoinFlipMaker(output_dimensions, p_replace, only_zero_flips=only_zero_flips)

    self.channels_dimension = self._agent.stack_size if self.share_dqn_conv else 1
    with tf.device(tf_device), tf.name_scope('intrinsic_coinflip'):
      if self.continuous_control:
        pre_processed_obs_shape = (None,) + observation_shape
        observation_shape = (1,) + observation_shape
      else:
        observation_shape = (1,) + observation_shape + (self.channels_dimension,)
        pre_processed_obs_shape = (None,) + self.resize_shape + (self.channels_dimension, )
      
      print(f'CFN: obs shape: {observation_shape} processed obs shape: {pre_processed_obs_shape}')
      
      obs_ph_type = tf.float32 if self.continuous_control else tf.uint8
      self.obs_ph = tf.placeholder(obs_ph_type, shape=observation_shape,
                                   name='obs_ph')
      self.preproccessed_obs = self._preprocess(self.obs_ph, resize_shape)
      self.preproccessed_obs_ph = tf.placeholder(obs_ph_type, shape=pre_processed_obs_shape,
                                   name='preprocess_obs_ph')
      self.preproccessed_next_obs_ph = tf.placeholder(obs_ph_type, shape=pre_processed_obs_shape,
                                   name='preprocess_next_obs_ph')
      self.coin_flip_targets = tf.placeholder(tf.float32, shape=[None, self.output_dimensions], name="coin_flip_targets")
      action_type = tf.float32 if self.continuous_control else tf.int32
      self.action_shape = (None, self.n_action_dims) if self.continuous_control else (None,)
      self.actions_ph = tf.placeholder(action_type, shape=self.action_shape,
                                   name='actions_ph')
      self.iter_ph = tf.placeholder(tf.uint32, shape=[], name='iter_num')
      self.iter = tf.cast(self.iter_ph, tf.float32)
      self.eval_ph = tf.placeholder(tf.bool, shape=[], name='eval_mode')
      self.prior_mean = tf.Variable(tf.zeros(shape=[1,self.output_dimensions]),
                                      trainable=False,
                                      name='prior_mean',
                                      dtype=tf.float32)
      # TODO: Possibly tune initialization...
      self.prior_var = tf.Variable(0.002 * tf.ones(shape=[1,self.output_dimensions]),
                                    trainable=False,
                                    name='prior_var',
                                    dtype=tf.float32)

      # For reward normalization in CFN
      self.reward_mean = tf.Variable(tf.zeros(shape=[]),
                                     trainable=False,
                                     name='reward_mean',
                                     dtype=tf.float32)
      self.reward_var = tf.Variable(tf.ones(shape=[]),
                                    trainable=False,
                                    name='reward_var',
                                    dtype=tf.float32)
      self.observation_mean = tf.Variable(tf.zeros(shape=observation_shape),
                                     trainable=False,
                                     name='observation_mean',
                                     dtype=tf.float32)
      self.observation_var = tf.Variable(tf.ones(shape=observation_shape), # should this be a better constant?
                                    trainable=False,
                                    name='observation_var',
                                    dtype=tf.float32)

      self.network = tf.make_template('FixedPointCountingNetwork', self._coin_flip_network_template)
      # self.ipd = tf.cond(tf.logical_not(self.eval_ph),
      #                    self.update,
      #                    self.virtual_update)
      update_dict = self.update()
      self.update_op = update_dict['train_op']
      self.one_over_counts_op = update_dict['one_over_counts']
      self.pred_coin_flip_op = update_dict['flips']
      self.prior_coin_flips_op = update_dict['prior_coin_flips']
      self.loss_op = update_dict['total_loss']
      self.coin_flip_loss_op = update_dict['coin_flip_loss']
      self.lwm_loss_op = update_dict['lwm_loss']
      self.count_consistency_loss_op = update_dict['count_consistency_loss']
      self.icm_loss_op = update_dict['icm_loss']
      self.lwm_output_op = update_dict['lwm_output']  # For logging only
      self.whitening_matrix_op = update_dict['whitening_matrix']
      if self.use_random_prior or self.use_reward_normalization or self.use_observation_normalization:
        self.normalizing_ops = self.make_normalizing_ops()
      self.reward = self.make_reward()
      self._online_summary_ops = self._make_online_summary_ops()

      print(f'Created CFN RewardLearner with n_action_dims={self.n_action_dims}')

  def make_cfn_replay_buffer(self):
    if self.share_dqn_conv:
      buffer_obs_shape = self.resize_shape + (self._agent.stack_size,)
    else:
      buffer_obs_shape = self.resize_shape
    env_dict = dict(
      obs=dict(
        shape = buffer_obs_shape,
        dtype=np.uint8
      ),
      coin_flip=dict(
        shape=(self.output_dimensions,),
        dtype=np.float32
      ),
      act=dict(
        # shape=(1,),
        dtype=np.int32),
    )

    # Create and return the CPPRB buffer
    if self.use_prioritized_buffer:
      return PrioritizedReplayBuffer(
        self.intrinsic_replay_buffer_size,
        env_dict,
        next_of=("obs",),
        stack_compress="obs" if self.share_dqn_conv else None,
        Nstep=False,
        alpha=1.0,
        eps=0.,
      )
    
    return ReplayBuffer(
      self.intrinsic_replay_buffer_size,
      env_dict,
      next_of=("obs",),
      stack_compress="obs" if self.share_dqn_conv else None,
      Nstep=False
    )

  def cc_make_cfn_replay_buffer(self):
    env_dict = dict(
      obs=dict(
        shape=self.observation_shape,
        dtype=np.float32
      ),
      coin_flip=dict(
        shape=(self.output_dimensions,),
        dtype=np.float32,
      ),
      act=dict(
        shape=(self.n_action_dims,),
        dtype=np.float32
      )
    )

    if self.use_prioritized_buffer:
      return PrioritizedReplayBuffer(
        self.intrinsic_replay_buffer_size,
        env_dict,
        next_of=("obs",),
        Nstep=False,
        alpha=1.0,
        eps=0.,
      )

    return ReplayBuffer(
      self.intrinsic_replay_buffer_size,
      env_dict,
      next_of=("obs",),
      Nstep=False
    )

  def _increment_num_updates(self, sample):
    # _store is the data store for memory.

    indices = sample["indexes"]
    self.num_updates_buffer.increment_priorities(indices)

  def _make_online_summary_ops(self):
    network = self.network(self.preproccessed_obs)
    trained_flips = network.trained_coin_flips
    trained_magnitude = tf.math.sqrt(tf.reduce_mean(tf.square(trained_flips), axis=1))
    full_flips = network.coin_flips
    unscaled_bonus = tf.reduce_mean(tf.square(full_flips), axis=1) ** self.bonus_exponent
    scaled_bonus = self.reward_scale * (unscaled_bonus - self.reward_mean) / self.reward_var
    if self.use_random_prior:
      full_magnitude = tf.math.sqrt(tf.reduce_mean(tf.square(full_flips), axis=1))
      prior_flips = network.prior_coin_flips
      prior_magnitude = tf.math.sqrt(tf.reduce_mean(tf.square(prior_flips), axis=1))
      rms_prior_mean = tf.math.sqrt(tf.reduce_mean(tf.square(self.prior_mean), axis=1))
      rms_prior_var = tf.math.sqrt(tf.reduce_mean(tf.square(self.prior_var), axis=1))
      return {
        'full_magnitude': full_magnitude,
        'prior_magnitude': prior_magnitude,
        'trained_magnitude': trained_magnitude,
        'rms_prior_mean': rms_prior_mean,
        'rms_prior_var': rms_prior_var,
        'unscaled_reward_mean': self.reward_mean,
        'unscaled_reward_var': self.reward_var,
        'unscaled_bonus' : unscaled_bonus,
        'scaled_bonus': scaled_bonus,
      }

    else:
      return {
        'trained_magnitude': trained_magnitude,
        'unscaled_bonus': unscaled_bonus,
        'scaled_bonus': scaled_bonus,
      }

  def make_normalizing_ops(self):
    """Update moment function passed later to a tf.cond."""
    assert self.use_random_prior or self.use_reward_normalization, "don't call this without random prior or reward_normalization"
    
    network = self.network(self.preproccessed_obs)
    unscaled_prior_coin_flips = network.unscaled_prior_coin_flips
    unnormalized_unscaled_reward = tf.reduce_mean(tf.square(network.coin_flips), axis=1) ** self.bonus_exponent
    unnormalized_unscaled_reward = tf.squeeze(unnormalized_unscaled_reward)

    effective_iter = self.iter - self.intrinsic_replay_start_size - 1 # I don't know why minus 1, but first one you see is 1002

    n_ops = 4 if self.use_reward_normalization and self.use_random_prior else 2
    if self.use_observation_normalization:
      n_ops += 2

    def _update():
      moments = []
      if self.use_reward_normalization:
        moments.append((unnormalized_unscaled_reward, self.reward_mean, self.reward_var))
      if self.use_random_prior:
        moments.append((unscaled_prior_coin_flips, self.prior_mean, self.prior_var))
      if self.use_observation_normalization:
        if self.continuous_control:
          preproccessed_obs = self.preproccessed_obs
        else:  # convert the uint8 image to a float to apply normalization
          preproccessed_obs = tf.cast(self.preproccessed_obs, tf.float32) / 255.
        moments.append((preproccessed_obs, self.observation_mean, self.observation_var))



      ops = []
      for value, mean, var in moments:
        delta = value - mean
        assign_mean = mean.assign_add(delta / effective_iter)
        var_ = var * effective_iter + (delta ** 2) * effective_iter / (effective_iter + 1)
        assign_var = var.assign(var_ / (effective_iter + 1))
        ops.extend([assign_mean, assign_var])

      return ops

    return tf.cond(
      tf.logical_not(self.eval_ph),
      lambda: _update(),
      # false_fn must have the same number and type of outputs.
      lambda: n_ops * [tf.constant(0., tf.float32)])

  def make_reward(self):
    # Will be something like 1 x 100 in shape. 
    # We want the RMS of it.
    # For various reasons, this is always receiving something that has a batch dimension.
    # So we need to squeeze it.

    network = self.network(self.preproccessed_obs)
    flips = network.coin_flips

    mean_square_flips = tf.reduce_mean(tf.square(flips), axis=1)
    magnitude = mean_square_flips ** self.bonus_exponent

    if self.use_random_prior or self.use_reward_normalization:  # with tf.control_dependencies(self._update_moments()):
      with tf.control_dependencies(self.normalizing_ops):
        reward = tf.squeeze(magnitude, axis=0)
    else:
      reward = tf.squeeze(magnitude, axis=0)

    return tf.cond(
      tf.logical_not(self.eval_ph),
      lambda: self.reward_scale * (reward - self.reward_mean) / self.reward_var,
      lambda: self.reward_scale * reward  # (eval-mode) Unnormalized rewards for plotting true vs approx bonuses
    )

  def make_batch_reward(self, input_tensor):
    # From my commits. Output of network will be [batch_size x num_flips]
    resized_input_tensor = self._preprocess(input_tensor, self.resize_shape)
    network = self.network(resized_input_tensor)
    flips = network.coin_flips
    magnitude = tf.reduce_mean(tf.square(flips), axis=1) ** self.bonus_exponent

    reward = self.reward_scale * magnitude
    reward = reward - self.reward_mean / self.reward_var
    return tf.stop_gradient(reward)


  def _add_summary_value(self, name, val, t=None):
    if t is None:
      t = self._t
    self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val)]), t)

  def _convert_obs(self, obs):
    if self.continuous_control:
      assert len(obs.shape) == 1, obs.shape
      return obs[np.newaxis, :]
    elif self.share_dqn_conv:
      # assert obs.shape == (1, 30), obs.shape
      return obs # (1,84, 84, 4) or (1, D)
    else:
      return obs[np.newaxis, :, :, np.newaxis]

  # @timing_wrapper
  def compute_intrinsic_reward(self, observation, training_steps, eval_mode, action=None):
    """Updates the model (during training), returns the intrinsic reward.

    Args:
      observation: Input frame. For compatibility with other models, this
        may have a batch-size of 1 as its first dimension.
      training_steps: Number of training steps, int.
      eval_mode: bool, whether or not running eval mode.
      Eval means test.

    Returns:
      The corresponding intrinsic reward.
    """
    # Adds batch size of 1.
    """
    We're going to do a bad thing here. self._agent.state is what we want to
    act from. That means we're just totally going to ignore the observation here.
    Not a huge fan of that but such is life.
    It'll mainly mess up the bonus plots for griddyboi.
    """
    if self.share_dqn_conv: # Not great!
      observation = self._agent.state

    if not eval_mode: # if training.
      coin_flip = self.coin_flip_maker()
      padded_observation = self._convert_obs(observation)
      preprocessed_obs_np = self._sess.run(self.preproccessed_obs, {
        self.obs_ph: padded_observation
      })

      if self.previous_obs is not None:
        self.add_to_buffer(self.previous_obs, preprocessed_obs_np, coin_flip, self.last_action)
      
      # Keeping track of current and previous obs for CPPRB
      self.previous_obs = preprocessed_obs_np
      self.last_action = action
      
      self._t += 1
      if self.replay_buffer.get_stored_size() > self.intrinsic_replay_start_size:
        # We want to start updating the random prior once we start training
        if self.replay_buffer.get_stored_size() <= self.intrinsic_replay_reward_add_start_size:
          self._sess.run(self.normalizing_ops, {self.obs_ph: padded_observation,
                                                    self.iter_ph: training_steps,
                                                    self.eval_ph: eval_mode}) # update normalizing ops
        if self._t % self.update_period == 0:
          self.count_estimator_learning_step()


    if self._t % 50 == 0 and not eval_mode and self.summary_writer is not None:
      logging_observation = self._convert_obs(observation)
      online_summary_ops_evaluated = self._sess.run(self._online_summary_ops, {self.obs_ph: logging_observation,
                                          self.iter_ph: training_steps,
                                          self.eval_ph: eval_mode})
      # Leaving all the 'magnitudes' with sqrts because that's what magnitude means
      # Adding an `online_unscaled_bonus` so we can see that thing.
      self._add_summary_value('CFN/trained_magnitude', online_summary_ops_evaluated['trained_magnitude'])
      self._add_summary_value('CFN/online_unscaled_bonus', online_summary_ops_evaluated['unscaled_bonus'])
      self._add_summary_value('CFN/online_scaled_bonus', online_summary_ops_evaluated['scaled_bonus'])
      if self.use_random_prior:
        self._add_summary_value('CFN/prior_magnitude', online_summary_ops_evaluated['prior_magnitude'])
        self._add_summary_value('CFN/full_magnitude', online_summary_ops_evaluated['full_magnitude'])
        self._add_summary_value('CFN/rms_prior_mean', online_summary_ops_evaluated['rms_prior_mean'])
        self._add_summary_value('CFN/rms_prior_var', online_summary_ops_evaluated['rms_prior_var'])
        self._add_summary_value('CFN/unscaled_reward_mean', online_summary_ops_evaluated['unscaled_reward_mean'])
        self._add_summary_value('CFN/unscaled_reward_var', online_summary_ops_evaluated['unscaled_reward_var'])

    if self.replay_buffer.get_stored_size() > self.intrinsic_replay_reward_add_start_size:
      # If fresh rewards, don't return the reward. But we still need to normalize.
      # Only normalize when eval is true
      padded_observation = self._convert_obs(observation)
      if self.use_fresh_rewards and not eval_mode:
        if self.use_random_prior or self.use_reward_normalization:
          self._sess.run(self.normalizing_ops, {self.obs_ph: padded_observation,
                                                self.iter_ph: training_steps,
                                                self.eval_ph: eval_mode})
        return 0.0
      else:
        return self._sess.run(self.reward, {self.obs_ph: padded_observation,
                                            self.iter_ph: training_steps,
                                            self.eval_ph: eval_mode})
    else:
      # Output isn't useful yet.
      return 0.

  def count_estimator_learning_step(self):
    # print(f'happening, batch size {self.batch_size}')
    """ Perform one step of gradient update on the coinflip approximator. """

    sample = self.replay_buffer.sample(self.batch_size)

    # Extract state and coin flips from sampled batch
    if self.continuous_control:
      expected_shape = (self.batch_size, self.observation_shape[0])
    elif self.share_dqn_conv:
      expected_shape = (self.batch_size,) + self.resize_shape + (self.channels_dimension,)
    else:
      expected_shape = (self.batch_size,) + self.resize_shape
    assert sample["obs"].shape == expected_shape, sample["obs"].shape
    
    if self.continuous_control or self.share_dqn_conv:
      preprocessed_obs_batch = sample["obs"]
    else:
      preprocessed_obs_batch = sample["obs"][..., np.newaxis]

    coin_flip_batch = sample["coin_flip"]
    action_batch = sample["act"]
    if self.continuous_control:
      assert action_batch.shape[0] == self.batch_size, action_batch.shape
      assert len(action_batch.shape) == 2, action_batch.shape
    else:
      assert len(action_batch.shape) == 2, f"action batch shape: {action_batch.shape}"
      action_batch = action_batch[:, 0]

    to_run = {
      'one_over_counts' : self.one_over_counts_op,
      'mse' : self.loss_op,
      'coin_flip_mse': self.coin_flip_loss_op,
      'pred_coin_flip': self.pred_coin_flip_op,
      'update_op' : self.update_op,
      'whitening_matrix' : self.whitening_matrix_op,
    }
    if self.use_random_prior:
      to_run['prior_coin_flips'] = self.prior_coin_flips_op

    feed_dict = {
      self.preproccessed_obs_ph: preprocessed_obs_batch,
      self.coin_flip_targets: coin_flip_batch,
      self.actions_ph: action_batch,
    }

    # TODO: not right for cc
    if self.use_lwm_representation_learning:
      feed_dict[self.preproccessed_next_obs_ph] = sample["next_obs"][..., np.newaxis]
      to_run['lwm_mse'] = self.lwm_loss_op
      to_run['lwm_output'] = self.lwm_output_op
    if self.use_count_consistency:
      if self.continuous_control:
        feed_dict[self.preproccessed_next_obs_ph] = sample["next_obs"]
      else:
        feed_dict[self.preproccessed_next_obs_ph] = sample["next_obs"][..., np.newaxis]
      to_run['count_consistency_mse'] = self.count_consistency_loss_op
    if self.use_icm_representation_learning:
      feed_dict[self.preproccessed_next_obs_ph] = sample["next_obs"][..., np.newaxis]
      to_run['icm_loss'] = self.icm_loss_op


    results = self._sess.run(to_run, feed_dict=feed_dict)

    if self.use_prioritized_buffer:
      self.update_priorities(sample, results["one_over_counts"])

    if self.summary_writer is not None:
      if self._t % 50 == 0:
        self._add_summary_value("CFN/mse", results["mse"])
        self._add_summary_value("CFN/coin_flip_mse", results["coin_flip_mse"])
        # NOTE: this was previously wrong. Was doing sqrt outside mean, should have been opposite.
        # self._add_summary_value("CFN/unscaled_bonus", np.sqrt(results["one_over_counts"]).mean())
        self._add_summary_value("CFN/buffer_unscaled_bonus", np.mean(results["one_over_counts"] ** self.bonus_exponent))
        self._add_summary_value("CFN/buffer_unscaled_bonus_variance", np.var(results["one_over_counts"] ** self.bonus_exponent))
        self._add_summary_value("CFN/mean_pred_coin_flip", np.mean(results["pred_coin_flip"]))
        self._add_summary_value("CFN/var_pred_coin_flip", np.var(results["pred_coin_flip"]))
        if self.use_random_prior:
          # self._add_summary_value("CFN/bonus_from_prior", np.sqrt(np.mean(results["prior_coin_flips"]**2)))
          bonus_from_prior = np.mean(np.mean(results["prior_coin_flips"]**2, axis=1) ** self.bonus_exponent)
          self._add_summary_value("CFN/bonus_from_prior", bonus_from_prior)
        if self.use_lwm_representation_learning:
          self._add_summary_value("CFN/lwm_mse", results["lwm_mse"])
          self._add_summary_value("CFN/lwm_output_norm", np.linalg.norm(results["lwm_output"], axis=1).mean())
          lwm_per_dim_mean_mag = np.linalg.norm(results["lwm_output"].mean(axis=0))
          lwm_per_dim_var_mag = np.linalg.norm(results["lwm_output"].var(axis=0))
          self._add_summary_value("CFN/lwm_per_dim_mean_mag", lwm_per_dim_mean_mag)
          self._add_summary_value("CFN/lwm_per_dim_var_mag", lwm_per_dim_var_mag)
          self._add_summary_value("CFN/lwm_whitening_frob_norm", np.linalg.norm(results["whitening_matrix"]).mean())
        if self.use_count_consistency:
          self._add_summary_value("CFN/count_consistency_mse", results["count_consistency_mse"])
        if self.use_icm_representation_learning:
          self._add_summary_value("CFN/icm_loss", results["icm_loss"])

  def update_priorities(self, sample, one_over_counts):
      indices = sample["indexes"]  # indices is usually at -2 for Non prioritized buffers
      num_updates = self.num_updates_buffer.sample(indices)
      assert one_over_counts.shape == num_updates.shape, (one_over_counts.shape, num_updates.shape)

      if self.prioritization_strategy == "combination":
        assert self.priority_alpha >= 0. and self.priority_alpha <= 1.
        new_priorities = (self.priority_alpha / (num_updates + 1)) + (
          1 - self.priority_alpha) * one_over_counts
      else:
        raise ValueError(f"prioritization strategy {self.prioritization_strategy} not supported")
      
      self.replay_buffer.update_priorities(indices, new_priorities)
      if self.prioritization_strategy == "combination":
        self._increment_num_updates(sample)

  def add_to_buffer(self, preprocessed_obs, preprocessed_next_obs, coin_flip, action):
    if self.use_prioritized_buffer:
      # Max priority
      priorities = 1.
      if self.prioritization_strategy == "combination":
        num_updates = 1
        self.replay_buffer.add(
            obs=preprocessed_obs[0],
            next_obs=preprocessed_next_obs[0],
            coin_flip=coin_flip,
            act=action,
            num_updates=num_updates,
            priorities=priorities
        )
      else:
        self.replay_buffer.add(
            obs=preprocessed_obs[0],
            next_obs=preprocessed_next_obs[0],
            coin_flip=coin_flip,
            act=action,
            priorities=priorities
        )
      self.num_updates_buffer.add(num_updates=num_updates)

    else:
      self.replay_buffer.add(
          obs=preprocessed_obs[0],
          next_obs=preprocessed_next_obs[0],
          coin_flip=coin_flip,
          act=action,
        )

  def _preprocess(self, obs, obs_shape):
    """
    Preprocess the input. Normalizes to [0,1]. NOT [0,8] like it was before
    """
    if self.continuous_control:
      assert obs.dtype == tf.float32, obs.dtype
      return obs

    assert obs.dtype == tf.uint8, obs.dtype
    obs = tf.cast(obs, tf.float32)
    obs = tf.image.resize_bilinear(obs, obs_shape)
    obs = tf.cast(obs, tf.uint8)
    return obs

  def _get_conv_info(self, layer_number, size="large"):
    conv_info = {
      'large' : {
        0: {"kernel" : (8, 8), "stride" : 4},
        1: {"kernel" : (4, 4), "stride" : 2},
        2: {"kernel" : (3, 3), "stride" : 1}
      },
      # Medium is probably appropriate for downsampling to 42x42.
      'medium' : {
        0: {"kernel" : (4, 4), "stride" : 2},
        1: {"kernel" : (4, 4), "stride" : 2},
        2: {"kernel" : (3, 3), "stride" : 1}
      },
      # Probably appropriate for 11 x 8
      'small' : {
        0: {"kernel" : (3, 3), "stride" : 1},
        1: {"kernel" : (3, 3), "stride" : 1},
        2: {"kernel" : (3, 3), "stride": 1}
      }
    }

    return conv_info[size][layer_number]

  def _prior_coin_flip_network_function(self, obs, n_conv_layers, kernel_sizes="large"):
    # NOTE: Assume we did the normalization already
    assert obs.dtype == tf.float32, obs.dtype
    assert kernel_sizes in ['small', 'medium', 'large']

    net = obs
    assert n_conv_layers in (0, 1, 2, 3), n_conv_layers
    final_activation = tf.math.tanh if self.use_final_tanh else None

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        trainable=False,
                        biases_initializer=tf.zeros_initializer()):
      if self.continuous_control:
        net = self._cc_prior_network_torso(net)
      else:
        net = self._conv_prior_torso(net, n_conv_layers, kernel_sizes)

      # This doesn't need another layer, eg RND doesn't have one
      prior_coin_flips = slim.fully_connected(net, self.output_dimensions, activation_fn=final_activation)
      unscaled_prior_coin_flips = prior_coin_flips
      # Doing scaling
      prior_coin_flips = (prior_coin_flips - self.prior_mean) / (tf.sqrt(self.prior_var + 0.0001))
      prior_coin_flips = tf.stop_gradient(prior_coin_flips)
      unscaled_prior_coin_flips = tf.stop_gradient(unscaled_prior_coin_flips)
      self.prior_coin_flips = prior_coin_flips
      self.unscaled_prior_coinflips = unscaled_prior_coin_flips

    return prior_coin_flips, unscaled_prior_coin_flips

  def _conv_prior_torso(self, net, n_conv_layers, kernel_sizes):
    if n_conv_layers > 0:
        print("did first conv layer")
        layer_info = self._get_conv_info(0, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = slim.conv2d(net, 32, kernel, stride=stride,
                          activation_fn=tf.nn.leaky_relu)
    if n_conv_layers > 1:
      print("did second conv layer")
      layer_info = self._get_conv_info(1, kernel_sizes)
      kernel, stride = layer_info['kernel'], layer_info['stride']
      net = slim.conv2d(net, 64, kernel, stride=stride,
                        activation_fn=tf.nn.leaky_relu)
    if n_conv_layers > 2:
      print("did third conv layer")
      layer_info = self._get_conv_info(2, kernel_sizes)
      kernel, stride = layer_info['kernel'], layer_info['stride']
      net = slim.conv2d(net, 64, kernel, stride=stride,
                        activation_fn=tf.nn.leaky_relu)

      net = slim.flatten(net) # Should have size 2304 when we resize to 42x42.
      return net
  
  def _cc_prior_network_torso(self, net):
    net = slim.fully_connected(net, 400, scope='prior_pred_fc1', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net, 300, scope='prior_pred_fc2', activation_fn=tf.nn.leaky_relu)
    return net

  @gin.configurable
  def _coin_flip_network_template(self, obs, n_conv_layers, kernel_sizes="large", fc_hidden=False, fc_hidden_size=512, stop_conv_grad=False):
    """
    Our network architecture. Will steal mostly from PixelCNN?
    Maybe I won't do that. Maybe I'll do my own architecture

    We get essentially the DQN architecture if we use a hidden FC layer. We get
    the RND architecture if we don't use a hidden layer.
    https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
    if input is 84x84x1:
      if n_conv_layers == 0: linear has 84x84=7000 dim
      if n_conv_layers == 1: linear has 7000*32/16=14000 dim
      if n_conv_layers == 2: linear has 14000*2/4=7000 dim
      if n_conv_layers == 3: linear has 7000*1/1=7000 dim
    """
    assert not (self.continuous_control and fc_hidden), "for now no fc_hidden along with continuous control"
    assert kernel_sizes in ['small', 'medium', 'large']
    if not self.continuous_control:
      assert obs.dtype == tf.uint8, obs.dtype
      obs = tf.cast(obs, tf.float32) # cast to float
      obs = obs / 255. # instead of in preprocess.
    if self.use_observation_normalization:
      obs = (obs - self.observation_mean) / (self.observation_var + 1e-6) # TODO: Not sure if the 1e-6 is necessary
      obs = tf.clip_by_value(obs, clip_value_min=-5., clip_value_max=5.)  # Same as RND

    net = obs

    assert n_conv_layers in (0, 1, 2, 3), n_conv_layers
    final_activation = tf.math.tanh if self.use_final_tanh else None
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      # Feature extraction
      if self.continuous_control:
        net = self._cc_network_torso(obs)
      else:
        net = self._conv_network_torso(obs, n_conv_layers, kernel_sizes, stop_conv_grad)

      if fc_hidden:
        net = slim.fully_connected(net, fc_hidden_size, scope='fc_hidden', activation_fn=tf.nn.leaky_relu)

      coin_flips = slim.fully_connected(net, self.output_dimensions, scope='fc_final', activation_fn=final_activation)
      trained_coin_flips = coin_flips
      if self.use_lwm_representation_learning:
        if self.shared_representation_learning_latent:
          # Same output layer!
          lwm_output = coin_flips
        else:
          lwm_output = slim.fully_connected(net, 32, scope='lwm_linear', activation_fn=None)
      else:
        lwm_output = None

    if self.use_random_prior:
      prior_coin_flips, unscaled_prior_coinflips = self._prior_coin_flip_network_function(obs, n_conv_layers, kernel_sizes=kernel_sizes)
      # TODO: Normalization nonsense
      coin_flips = coin_flips + prior_coin_flips
    else:
      prior_coin_flips, unscaled_prior_coinflips = None, None

    self.coin_flips = coin_flips
    loss = tf.losses.mean_squared_error(
      self.coin_flip_targets, coin_flips,
      reduction=tf.losses.Reduction.MEAN)

    return collections.namedtuple('CoinFlip_network',
        ['coin_flips', 'loss', 'trained_coin_flips', 'prior_coin_flips', 'unscaled_prior_coin_flips', 'lwm_output', 'last_layer_output'])(
        coin_flips, loss, trained_coin_flips, prior_coin_flips, unscaled_prior_coinflips, lwm_output, net)

  def _cc_network_torso(self, obs):
    net = slim.fully_connected(obs, 400, scope='pred_fc1', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net, 300, scope='pred_fc2', activation_fn=tf.nn.leaky_relu)
    return net

  def _conv_network_torso(self, net, n_conv_layers, kernel_sizes, stop_conv_grad):
    if n_conv_layers > 0:
      print("did first conv layer")
      if self.share_dqn_conv:
        net = self._agent.online_convnet.conv1(net)
      else:
        layer_info = self._get_conv_info(0, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = slim.conv2d(net, 32, kernel, scope='conv1', stride=stride,
                          activation_fn=tf.nn.leaky_relu)
      self.conv1_output = net
    if n_conv_layers > 1:
      print("did second conv layer")
      if self.share_dqn_conv:
        net = self._agent.online_convnet.conv2(net)
      else:
        layer_info = self._get_conv_info(1, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = slim.conv2d(net, 64, kernel, scope='conv2', stride=stride,
                          activation_fn=tf.nn.leaky_relu)
      self.conv2_output = net
    if n_conv_layers > 2:
      print("did third conv layer")
      if self.share_dqn_conv:
        net = self._agent.online_convnet.conv3(net)
      else:
        layer_info = self._get_conv_info(2, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = slim.conv2d(net, 64, kernel, scope='conv3', stride=stride,
                          activation_fn=tf.nn.leaky_relu)
      self.conv3_output = net

    net = slim.flatten(net) # Should have size 2304 when we resize to 42x42.

    if stop_conv_grad:
      net = tf.stop_gradient(net)
    
    return net

  def get_whitening_params(self, lwm_out_1, lwm_out_2):
    num_features = self.output_dimensions if self.shared_representation_learning_latent else 32
    identity_matrix = tf.eye(num_features)
    def smoothen_matrix(X, I, eps=1e-4):
      return (eps * I)  + ((1-eps) * X)
    lwm_concat = tf.concat([lwm_out_1, lwm_out_2], axis=0)
    lwm_mean = tf.reduce_mean(lwm_concat, axis=0)[None,...] # batch dimension
    covariance_matrix = tfp.stats.covariance(lwm_concat)
    smooth_cov_matrix = smoothen_matrix(covariance_matrix, identity_matrix)
    L = tf.linalg.cholesky(smooth_cov_matrix)  # A = LL' (L lower triangular)
    L_inv = tf.linalg.triangular_solve(L, identity_matrix, lower=True)  # Same as W (whitening matrix)
    return lwm_mean, tf.transpose(L_inv)  # Its going to be V W.T

  def update(self):
    """
    I guess we expect that we'll have our coin_flip placeholder filled here?
    Or, possibly we just fetch a batch and do an update? We can handle the
    `compute_intrinsic_reward` separately I guess.

    I think that's actually better. 
    """
    with tf.name_scope('update'):
      network_output = self.network(self.preproccessed_obs_ph)
      next_network_output = self.network(self.preproccessed_next_obs_ph)
      flips = network_output.coin_flips
      next_flips = next_network_output.coin_flips
      prior_coin_flips = network_output.prior_coin_flips
      one_over_counts = tf.reduce_mean(tf.square(flips), axis=1)

      coin_flip_loss = network_output.loss
      lwm_out_1 = network_output.lwm_output
      lwm_out_2 = next_network_output.lwm_output
      if self.use_lwm_representation_learning:
        if self.use_representation_whitening:
          lwm_mean, whitening_matrix = self.get_whitening_params(lwm_out_1=lwm_out_1, lwm_out_2=lwm_out_2)
          lwm_out_1 = tf.matmul((lwm_out_1 - lwm_mean), whitening_matrix)
          lwm_out_2 = tf.matmul((lwm_out_2 - lwm_mean), whitening_matrix)
        lwm_loss = tf.losses.mean_squared_error(
          lwm_out_1, lwm_out_2,
          reduction=tf.losses.Reduction.MEAN,
        )
      else:
        lwm_loss = tf.constant(0.)
        whitening_matrix = tf.constant(0.)
      if self.use_count_consistency:
        one_over_next_counts = tf.reduce_mean(tf.square(next_flips), axis=1)
        count_consistency_loss = tf.losses.mean_squared_error(
          one_over_counts, one_over_next_counts,
          reduction=tf.losses.Reduction.MEAN,
        )
      else:
        count_consistency_loss = tf.constant(0.)
      if self.use_icm_representation_learning:
        # We need to make the representations here.
        # https://github.com/pathak22/noreward-rl/blob/3e220c2177fc253916f12d980957fc40579d577a/src/model.py#L236
        last_output = network_output.last_layer_output
        next_last_output = next_network_output.last_layer_output
        # Debatably we should do a conv here, but no for now.
        concatenated_last_outputs = tf.concat([last_output, next_last_output], axis=1)
        concatenated_hidden_layer = slim.fully_connected(
          concatenated_last_outputs, 128, scope='action_prediction_hidden', activation_fn=tf.nn.leaky_relu)
        action_prediction_logits = slim.fully_connected(
          concatenated_hidden_layer, self.num_actions, scope='action_prediction_logits', activation_fn=None)
        if self.continuous_control:  # TODO
          raise NotImplementedError('Implement ICM for cc')
        icm_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels=self.actions_ph, logits=action_prediction_logits), name="icm_loss")
      else:
        icm_loss = tf.constant(0.)

        # inverse_count_this = tf.reduce_sum(network_output.coin_flips**2

      # total_loss = coin_flip_loss + (self.lwm_representation_learning_scale * lwm_loss) + (self.count_consistency_scale * count_consistency_loss)
      total_loss = coin_flip_loss
      total_loss += (self.lwm_representation_learning_scale * lwm_loss)
      total_loss += (self.count_consistency_scale * count_consistency_loss)
      total_loss += (self.icm_representation_learning_scale * icm_loss)

      # flips = network_output.coin_flips
      # prior_coin_flips = network_output.prior_coin_flips
      # one_over_counts = tf.reduce_mean(tf.square(flips), axis=1)
      train_op = self.optimizer.minimize(total_loss)
    return dict(
      train_op=train_op,
      one_over_counts=one_over_counts,
      flips=flips,
      prior_coin_flips=prior_coin_flips,
      coin_flip_loss=coin_flip_loss,
      lwm_loss=lwm_loss,
      total_loss=total_loss,
      lwm_output= lwm_out_1,
      whitening_matrix=whitening_matrix,
      count_consistency_loss=count_consistency_loss,
      icm_loss=icm_loss,
    )


@gin.configurable
class RNDIntrinsicReward(object):
  """Class used to instantiate a bonus using random network distillation."""

  def __init__(self,
               sess,
               embedding_size=512,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               tf_device='/gpu:0',
               reward_scale=1.0,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0001,
                   epsilon=0.00001),
               summary_writer=None,
               continuous_control=False):
    print('making the RND Intrinsic Reward!')
    self.embedding_size = embedding_size
    self.reward_scale = reward_scale
    self.optimizer = optimizer
    self._sess = sess
    self.summary_writer = summary_writer
    self.continuous_control = continuous_control
    self._t = 0

    with tf.device(tf_device), tf.name_scope('intrinsic_rnd'):
      if self.continuous_control:
        obs_shape = (1,) + observation_shape
      else:
        obs_shape = (1,) + observation_shape + (1,)
      self.iter_ph = tf.placeholder(tf.uint64, shape=[], name='iter_num')
      self.iter = tf.cast(self.iter_ph, tf.float32)
      if self.continuous_control:
        self.obs_ph = tf.placeholder(tf.float32, shape=obs_shape,
                                    name='obs_ph')
      else:
        self.obs_ph = tf.placeholder(tf.uint8, shape=obs_shape,
                                    name='obs_ph')
      self.eval_ph = tf.placeholder(tf.bool, shape=[], name='eval_mode')
      self.obs = tf.cast(self.obs_ph, tf.float32)
      # Placeholder for running mean and std of observations and rewards
      self.obs_mean = tf.Variable(tf.zeros(shape=obs_shape),
                                  trainable=False,
                                  name='obs_mean',
                                  dtype=tf.float32)
      self.obs_std = tf.Variable(tf.ones(shape=obs_shape),
                                 trainable=False,
                                 name='obs_std',
                                 dtype=tf.float32)
      self.reward_mean = tf.Variable(tf.zeros(shape=[]),
                                     trainable=False,
                                     name='reward_mean',
                                     dtype=tf.float32)
      self.reward_std = tf.Variable(tf.ones(shape=[]),
                                    trainable=False,
                                    name='reward_std',
                                    dtype=tf.float32)
      self.obs = self._preprocess(self.obs)
      self.target_embedding = self._target_network(self.obs)
      self.prediction_embedding = self._prediction_network(self.obs)
      self._train_op = self._build_train_op()

  def _preprocess(self, obs):
    return tf.clip_by_value((obs - self.obs_mean) / self.obs_std, -5.0, 5.0)

  def compute_intrinsic_reward(self, obs, training_step, eval_mode=False, action=None):
    """Computes the RND intrinsic reward."""
    if self.reward_scale == 0:
      return 0.0
    if self.continuous_control:
      obs = obs[np.newaxis, :]
    else:
      obs = obs[np.newaxis, :, :, np.newaxis]
    to_evaluate = [self.intrinsic_reward]
    if not eval_mode:
      self._t += 1
      # Also update the prediction network
      to_evaluate.append(self._train_op)
    reward = self._sess.run(to_evaluate,
                            {self.obs_ph: obs,
                             self.iter_ph: training_step,
                             self.eval_ph: eval_mode})[0]
    if not eval_mode:
      if self._t % 50 == 0:
        self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="RND/unscaled_intrinsic_reward", simple_value=float(reward))]), self._t)        
        self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="RND/scaled_intrinsic_reward", simple_value=self.reward_scale*float(reward))]), self._t)
        reward_mean, reward_var = self._sess.run([self.reward_mean, self.reward_std])
        self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="RND/reward_mean", simple_value=reward_mean)]), self._t)        
        self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="RND/reward_var", simple_value=reward_var)]), self._t)
    return self.reward_scale * float(reward)

  def _target_network_cc(self, obs):
    with slim.arg_scope([slim.fully_connected], trainable=False,
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.fully_connected(obs, 400, scope='target_fc1', activation_fn=tf.nn.leaky_relu)
      embedding = slim.fully_connected(net, self.embedding_size, scope='target_fc2', activation_fn=None)
      return embedding

  def _target_network(self, obs):
    """Implements the random target network used by RND."""
    if self.continuous_control:
      return self._target_network_cc(obs)
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=False,
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.conv2d(obs, 32, [8, 8], stride=4,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [4, 4], stride=2,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [3, 3], stride=1,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.flatten(net)
      embedding = slim.fully_connected(net, self.embedding_size,
                                       activation_fn=None)
    return embedding

  def _prediction_network_cc(self, obs):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.fully_connected(obs, 400, scope='pred_fc1', activation_fn=tf.nn.leaky_relu)
      net = slim.fully_connected(net, 300, scope='pred_fc2', activation_fn=tf.nn.leaky_relu)
      embedding = slim.fully_connected(net, self.embedding_size, scope='pred_fc3', activation_fn=None)
    return embedding



  def _prediction_network(self, obs):
    """Prediction network used by RND to predict to target network output."""
    if self.continuous_control:
      return self._prediction_network_cc(obs)

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.conv2d(obs, 32, [8, 8], stride=4,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [4, 4], stride=2,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [3, 3], stride=1,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.flatten(net)
      net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
      net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
      embedding = slim.fully_connected(net, self.embedding_size,
                                       activation_fn=None)
    return embedding

  def _update_moments(self):
    """Update the moments estimates, assumes a batch size of 1."""
    def update():
      """Update moment function passed later to a tf.cond."""
      moments = [
          (self.obs, self.obs_mean, self.obs_std),
          (self.loss, self.reward_mean, self.reward_std)
      ]
      ops = []
      for value, mean, std in moments:
        delta = value - mean
        assign_mean = mean.assign_add(delta / self.iter)
        std_ = std * self.iter + (delta ** 2) * self.iter / (self.iter + 1)
        assign_std = std.assign(std_ / (self.iter + 1))
        ops.extend([assign_mean, assign_std])
      return ops

    return tf.cond(
        tf.logical_not(self.eval_ph),
        update,
        # false_fn must have the same number and type of outputs.
        lambda: 4 * [tf.constant(0., tf.float32)])

  def _build_train_op(self):
    """Returns train op to update the prediction network."""
    prediction = self.prediction_embedding
    target = tf.stop_gradient(self.target_embedding)
    self.loss = tf.losses.mean_squared_error(
        target, prediction, reduction=tf.losses.Reduction.MEAN)
    with tf.control_dependencies(self._update_moments()):
      self.intrinsic_reward = (self.loss - self.reward_mean) / self.reward_std
    return self.optimizer.minimize(self.loss)
