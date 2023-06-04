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

"""Implementation of a Rainbow agent with intrinsic rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bonus_based_exploration.intrinsic_motivation import intrinsic_dqn_agent
from bonus_based_exploration.intrinsic_motivation import intrinsic_rewards
from dopamine.agents.dqn import dqn_agent as base_dqn_agent
from dopamine.agents.rainbow import rainbow_agent as base_rainbow_agent
from dopamine.agents.rainbow.rainbow_agent import project_distribution
from dopamine.discrete_domains import atari_lib
import gin
import tensorflow.compat.v1 as tf


class FreshRewardRainbowAgent(base_rainbow_agent.RainbowAgent):
  def __init__(self,*args, update_horizon=1, stack_size=4, use_fresh_rewards=False, **kwargs):
    # We're going to take the final frame of state, and then the rest from next-state.
    # So, we need update_horizon-1 from next-state. So, stack_size just needs to be bigger
    if use_fresh_rewards:
      assert stack_size >= update_horizon, f"stack size needs to be greater than update_horizon, got {stack_size}<{update_horizon}"
    self.use_fresh_rewards = use_fresh_rewards
    super(FreshRewardRainbowAgent, self).__init__(
      *args,
      update_horizon=update_horizon,
      stack_size=stack_size,
      **kwargs)

  def _build_fresh_reward(self):
    """Build the fresh reward.
    First, get all the states we're working with. Then, reshape, get intrinsic rewards, unshape, dot with gamma, return.

    Dimensions are [batch_size x w x h x stack_size] (https://github.com/google/dopamine/blob/master/dopamine/replay_memory/circular_replay_buffer.py#L177)
    """
    states = self._replay.states # [batch_size x  w x h x stack_size] maybe? I should check
    next_states = self._replay.next_states
    batch_size = self._replay.batch_size
    num_frames_from_next = self.update_horizon - 1 # one of them is from states.

    final_start_state = states[:,:,:,-1:] # keep the final dimension
    # If there was a huge stack size, it would go: [past..., final_start_state, update_horizon-1 more frames, rest...]
    # So, what's the index of final_start_state? The last in final_target_state is state update_horizon+1.
    # So, we want to go to the one before that. Meaning the last one we want to include is always the one before last.
    # We always expect it to go to the second-to-last (doesn't include the include the ).
    # if update_horizon was 1, then we don't use next_states.
    # If update_horizon was 2, then we use the second-to-last next-state.
    # If update_horizon was 3, then we use the third-to-last and second-to-last next-state.
    # In that case, if stack-size is 4, then that would be index -3 and -2, which is what this gives us.
    if num_frames_from_next > 0:
      end_states = next_states[:,:,:,-1-num_frames_from_next:-1]
      all_states = tf.concat([final_start_state, end_states], axis=3)
    else:
      all_states = final_start_state

    all_states_reshaped = tf.transpose(all_states, perm=[0,3,1,2])
    # combine first two dimensions, then add channel at end.
    all_states_reshaped = tf.reshape(all_states_reshaped, shape=[-1, self.observation_shape[0], self.observation_shape[1], 1])

    # Get the intrinsic rewards for all states, sum together
    intrinsic_rewards = self.intrinsic_model.make_batch_reward(all_states_reshaped)
    intrinsic_rewards = tf.reshape(intrinsic_rewards, shape=[batch_size, self.update_horizon])
    gammas = [self.gamma**i for i in range(self.update_horizon)]
    gammas = tf.convert_to_tensor(gammas)[None, ...] # Maybe need to put on GPU, I don't know.
    summed_intrinsic_rewards = tf.reduce_sum(intrinsic_rewards * gammas, axis=1, keepdims=True)

    return tf.stop_gradient(summed_intrinsic_rewards)

  def _build_target_distribution(self):
    """
    UNTOUCHED, except adding something if we use fresh rewards.
    Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    batch_size = self._replay.batch_size

    # size of rewards: batch_size x 1
    rewards = self._replay.rewards[:, None]
    if self.use_fresh_rewards: # Only new part
      rewards += self._build_fresh_reward()

    # size of tiled_support: batch_size x num_atoms
    tiled_support = tf.tile(self._support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

    # size of target_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    target_support = rewards + gamma_with_terminal * tiled_support

    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        self._replay_next_target_net_outputs.q_values, axis=1)[:, None]
    batch_indices = tf.range(tf.cast(batch_size, tf.int64))[:, None]
    # size of next_qt_argmax: batch_size x 2
    batch_indexed_next_qt_argmax = tf.concat(
        [batch_indices, next_qt_argmax], axis=1)

    # size of next_probabilities: batch_size x num_atoms
    next_probabilities = tf.gather_nd(
        self._replay_next_target_net_outputs.probabilities,
        batch_indexed_next_qt_argmax)

    return project_distribution(target_support, next_probabilities,
                                self._support)

@gin.configurable
class PixelCNNRainbowAgent(
    base_rainbow_agent.RainbowAgent,
    intrinsic_dqn_agent.PixelCNNDQNAgent):
  """A Rainbow agent paired with a pseudo count derived from a PixelCNN."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=base_dqn_agent.NATURE_DQN_DTYPE,
               stack_size=base_dqn_agent.NATURE_DQN_STACK_SIZE,
               network=atari_lib.RainbowNetwork,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0000625, epsilon=0.00015),
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
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
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
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.intrinsic_model = intrinsic_rewards.PixelCNNIntrinsicReward(
        sess=sess,
        tf_device=tf_device)
    super(PixelCNNRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def step(self, reward, observation):
    return intrinsic_dqn_agent.PixelCNNDQNAgent.step(
        self, reward, observation)


@gin.configurable
class CoinFlipRainbowAgent(
    FreshRewardRainbowAgent,
    intrinsic_dqn_agent.CoinFlipDQNAgent):
  """A Rainbow agent paired with a pseudo count derived from a PixelCNN."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=base_dqn_agent.NATURE_DQN_DTYPE,
               stack_size=base_dqn_agent.NATURE_DQN_STACK_SIZE,
               network=atari_lib.RainbowNetwork,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0000625, epsilon=0.00015),
               summary_writer=None,
               summary_writing_frequency=500,
               use_fresh_rewards=False):
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
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
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
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    # Moved to before super call. Note that "Agent" will be pretty unpopulated at start,
    # but luckily we don't need it much. Oh, we do if we want to share conv. Hmm.
    # With fresh-rewards, we need access to the intrinsic model so that we can include it in the target-value-comp.
    # But for share_conv, we need the agent first. Considering fresh-rewards doesn't work with share-conv,
    # this conditional is our current compromise 
    if use_fresh_rewards:
      self.intrinsic_model = intrinsic_rewards.CoinFlipCounterIntrinsicReward(
        sess=sess,
        tf_device=tf_device,
        summary_writer=summary_writer,
        agent=self,
        num_actions=num_actions,
        use_fresh_rewards=use_fresh_rewards)

    super(CoinFlipRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        use_fresh_rewards=use_fresh_rewards)

    if not use_fresh_rewards:
      self.intrinsic_model = intrinsic_rewards.CoinFlipCounterIntrinsicReward(
        sess=sess,
        tf_device=tf_device,
        summary_writer=summary_writer,
        agent=self,
        num_actions=num_actions,
        use_fresh_rewards=use_fresh_rewards)



  def step(self, reward, observation):
    return intrinsic_dqn_agent.CoinFlipDQNAgent.step(
        self, reward, observation)

@gin.configurable
class RNDRainbowAgent(
    base_rainbow_agent.RainbowAgent,
    intrinsic_dqn_agent.RNDDQNAgent):
  """A Rainbow agent paired with an intrinsic bonus derived from RND."""

  def __init__(self,
               sess,
               num_actions,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0000625, epsilon=0.00015),
               summary_writer=None,
               summary_writing_frequency=500,
               clip_reward=False):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
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
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      clip_reward: bool, whether or not clip the mixture of rewards.
    """
    self._clip_reward = clip_reward
    self.intrinsic_model = intrinsic_rewards.RNDIntrinsicReward(
        sess=sess,
        tf_device=tf_device,
        summary_writer=summary_writer)
    super(RNDRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _add_intrinsic_reward(self, observation, extrinsic_reward, action=None):
    return intrinsic_dqn_agent.RNDDQNAgent._add_intrinsic_reward(
        self, observation, extrinsic_reward, action=action)
