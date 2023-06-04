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

"""Implementation of a DQN agent with intrinsic rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

from bonus_based_exploration.intrinsic_motivation import base_intrinsic_agent
from bonus_based_exploration.intrinsic_motivation import intrinsic_rewards

from dopamine.agents.dqn import dqn_agent as base_dqn_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf



linearly_decaying_epsilon = base_dqn_agent.linearly_decaying_epsilon


@gin.configurable
class PixelCNNDQNAgent(base_intrinsic_agent.IntrinsicDQNAgent):
  """Implements a DQN agent with a RND intrinsic reward."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=base_dqn_agent.NATURE_DQN_DTYPE,
               stack_size=base_dqn_agent.NATURE_DQN_STACK_SIZE,
               network=base_dqn_agent.nature_dqn_network,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=3,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expecting 2 parameters: num_actions,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.intrinsic_model = intrinsic_rewards.PixelCNNIntrinsicReward(sess=sess)
    super(PixelCNNDQNAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)


@gin.configurable
class CoinFlipDQNAgent(base_intrinsic_agent.IntrinsicDQNAgent):
  """Implements a DQN agent with a CoinFlip intrinsic reward"""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=base_dqn_agent.NATURE_DQN_DTYPE,
               stack_size=base_dqn_agent.NATURE_DQN_STACK_SIZE,
               network=base_dqn_agent.nature_dqn_network,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=3,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expecting 2 parameters: num_actions,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.intrinsic_model = intrinsic_rewards.CoinFlipCounterIntrinsicReward(
      sess=sess,
      summary_writer=summary_writer,
      num_actions=num_actions)
    super(CoinFlipDQNAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def end_episode(self, reward):
    self.intrinsic_model.coin_flip_maker.reset()
    # Note that apparently they're smart enough to not break the indices,
    # so no changes to num_updates_buffer are needed
    to_return = super().end_episode(reward)
    # We need to add the final state to the buffer now.
    self.intrinsic_model.compute_intrinsic_reward(
      self._observation, self.training_steps, self.eval_mode, self.action)
    # And now we reset.
    self.intrinsic_model.previous_obs = None
    self.intrinsic_model.last_action = None
    self.intrinsic_model.replay_buffer.on_episode_end()
    return to_return


  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    # Make replay buffer checkpoint thing.
    intrinsic_buffer_dir = os.path.join(checkpoint_dir, 'intrinsic_buffer')
    os.makedirs(intrinsic_buffer_dir, exist_ok=True)
    intrinsic_buffer_filename = os.path.join(intrinsic_buffer_dir, f"buffer_{iteration_number}.npz")
    bundle_directory = super().bundle_and_checkpoint(checkpoint_dir, iteration_number)
    self.intrinsic_model.replay_buffer.save_transitions(intrinsic_buffer_filename)
    return bundle_directory

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    intrinsic_buffer_dir = os.path.join(checkpoint_dir, 'intrinsic_buffer')
    intrinsic_buffer_filename = os.path.join(intrinsic_buffer_dir, f"buffer_{iteration_number}.npz")
    try:
      self.intrinsic_model.replay_buffer.load_transitions(intrinsic_buffer_filename)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload INTRINSIC replay buffer!')

    return super().unbundle(checkpoint_dir, iteration_number, bundle_dictionary)


@gin.configurable
class RNDDQNAgent(base_intrinsic_agent.IntrinsicDQNAgent):
  """Implements a DQN agent with a RND intrinsic reward."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=3,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               clip_reward=False):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      clip_reward: bool, whether or not to clip the reward after adding
        the intrinsic reward.
    """
    self._clip_reward = clip_reward
    self.intrinsic_model = intrinsic_rewards.RNDIntrinsicReward(
        sess=sess,
        tf_device=tf_device,
        summary_writer=summary_writer)
    super(RNDDQNAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _add_intrinsic_reward(self, observation, extrinsic_reward, action=None):
    """Compute the intrinsic reward for RND."""
    intrinsic_reward = self.intrinsic_model.compute_intrinsic_reward(
        observation, self.training_steps, self.eval_mode, action=action)
    reward = extrinsic_reward + intrinsic_reward

    if self._clip_reward:
      intrinsic_reward = np.clip(intrinsic_reward, -1., 1.)
      reward = np.clip(reward, -1., 1.)
    if (self.summary_writer is not None and
        self.training_steps % self.summary_writing_frequency == 0):
      summary = tf.Summary(value=[
          tf.Summary.Value(tag='Train/ExtrinsicReward',
                           simple_value=extrinsic_reward),
          tf.Summary.Value(tag='Train/IntrinsicReward',
                           simple_value=intrinsic_reward),
          tf.Summary.Value(tag='Train/TotalReward',
                           simple_value=reward)
      ])
      self.summary_writer.add_summary(summary, self.training_steps)

    return float(reward)
