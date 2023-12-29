from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.jax.agents.sac import sac_agent as base_sac_agent
import numpy as np
import tensorflow.compat.v1 as tf
from bonus_based_exploration.intrinsic_motivation import intrinsic_rewards
from dopamine.jax import continuous_networks
import gin

tf.disable_eager_execution()

class BaseIntrinsicSACAgent(base_sac_agent.SACAgent):
  def _add_intrinsic_reward(self, observation, extrinsic_reward, action=None):
    """Compute the intrinsic reward."""
    if not hasattr(self, 'intrinsic_model'):
      raise NotImplementedError
    intrinsic_reward = self.intrinsic_model.compute_intrinsic_reward(
        observation, self.training_steps, self.eval_mode, action=action)
    reward = np.clip(intrinsic_reward + extrinsic_reward, -1., 1.)

    return reward

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    total_reward = self._add_intrinsic_reward(self._observation, reward, action=self.action)
    return base_sac_agent.SACAgent.step(self, total_reward, observation)

  def end_episode(self, reward, terminal=False):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    total_reward = self._add_intrinsic_reward(self._observation, reward, action=self.action)
    base_sac_agent.SACAgent.end_episode(self, total_reward, terminal=terminal)


@gin.configurable
class RNDSACAgent(BaseIntrinsicSACAgent):

  def __init__(self,
               action_shape,
               action_limits,
               observation_shape,
               action_dtype=np.float32,
               observation_dtype=np.float32,
               reward_scale_factor=1.0,
               stack_size=1,
               network=continuous_networks.SACNetwork,
               num_layers=2,
               hidden_units=256,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=1,
               target_update_type='soft',
               target_update_period=1000,
               target_smoothing_coefficient=0.005,
               target_entropy=None,
               eval_mode=False,
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False,
               seed=None,
               sess=None):
    # We'll have to do something different from "num_actions" here.
    # We'll also have to do something different for making the network. We'll also have to
    # do something different for computing the action-prediction-error. That's all fine. Will have to
    # happen inside of the CFCIR though.
    # Happens in ContinuousRunner instead

    # Whereas before, observation shape was set in config, now its more variable, so we pass through.
    self.intrinsic_model = intrinsic_rewards.RNDIntrinsicReward(
      sess=sess,
      summary_writer=summary_writer, # for now?
      observation_shape=observation_shape,
      # num_actions=num_actions,
      continuous_control=True)

    super(RNDSACAgent, self).__init__(
      action_shape=action_shape,
      action_limits=action_limits,
      observation_shape=observation_shape,
      action_dtype=action_dtype,
      observation_dtype=observation_dtype,
      reward_scale_factor=reward_scale_factor,
      stack_size=stack_size,
      network=network,
      num_layers=num_layers,
      hidden_units=hidden_units,
      gamma=gamma,
      update_horizon=update_horizon,
      min_replay_history=min_replay_history,
      update_period=update_period,
      target_update_type=target_update_type,
      target_update_period=target_update_period,
      target_smoothing_coefficient=target_smoothing_coefficient,
      target_entropy=target_entropy,
      eval_mode=eval_mode,
      optimizer=optimizer,
      summary_writer=None, # SACAgent assumes Flax, which breaks
      summary_writing_frequency=summary_writing_frequency,
      allow_partial_reload=allow_partial_reload,
      seed=seed)


@gin.configurable
class CoinFlipSACAgent(BaseIntrinsicSACAgent):
  """Implements a SAC agent with CFN intrinsic reward."""
  
  def __init__(self,
               action_shape,
               action_limits,
               observation_shape,
               action_dtype=np.float32,
               observation_dtype=np.float32,
               reward_scale_factor=1.0,
               stack_size=1,
               network=continuous_networks.SACNetwork,
               num_layers=2,
               hidden_units=256,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=1,
               target_update_type='soft',
               target_update_period=1000,
               target_smoothing_coefficient=0.005,
               target_entropy=None,
               eval_mode=False,
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False,
               seed=None,
               sess=None):
    # We'll have to do something different from "num_actions" here.
    # We'll also have to do something different for making the network. We'll also have to
    # do something different for computing the action-prediction-error. That's all fine. Will have to
    # happen inside of the CFCIR though.
    # Happens in ContinuousRunner instead

    n_dims = action_shape[0] if isinstance(action_shape, tuple) else action_shape
    assert isinstance(n_dims, int), f'{n_dims}, {type(n_dims)}'

    # Whereas before, observation shape was set in config, now its more variable, so we pass through.
    self.intrinsic_model = intrinsic_rewards.CoinFlipCounterIntrinsicReward(
      sess=sess,
      summary_writer=summary_writer,
      observation_shape=observation_shape,
      continuous_control=True,
      n_action_dims=n_dims)

    super(CoinFlipSACAgent, self).__init__(
      action_shape=action_shape,
      action_limits=action_limits,
      observation_shape=observation_shape,
      action_dtype=action_dtype,
      observation_dtype=observation_dtype,
      reward_scale_factor=reward_scale_factor,
      stack_size=stack_size,
      network=network,
      num_layers=num_layers,
      hidden_units=hidden_units,
      gamma=gamma,
      update_horizon=update_horizon,
      min_replay_history=min_replay_history,
      update_period=update_period,
      target_update_type=target_update_type,
      target_update_period=target_update_period,
      target_smoothing_coefficient=target_smoothing_coefficient,
      target_entropy=target_entropy,
      eval_mode=eval_mode,
      optimizer=optimizer,
      summary_writer=None, # SACAgent assumes Flax, which breaks
      summary_writing_frequency=summary_writing_frequency,
      allow_partial_reload=allow_partial_reload,
      seed=seed)
