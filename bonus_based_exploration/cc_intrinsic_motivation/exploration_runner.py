import os
import gin
import time
import pickle
import gzip
import numpy as np

from absl import logging
from copy import deepcopy
from collections import defaultdict
from flax.metrics import tensorboard
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains.checkpointer import CHECKPOINT_DURATION
from bonus_based_exploration.cc_intrinsic_motivation.intrinsic_SAC import BaseIntrinsicSACAgent
import tensorflow.compat.v1 as tf

def _safe_zip_write(filename, data):
  """Safely writes a file to disk.

  Args:
    filename: str, the name of the file to write.
    data: the data to write to the file.
  """
  filename_temp = f"{filename}.tmp.gz"
  with gzip.open(filename_temp, 'wb+') as f:
    pickle.dump(data, f)
  os.replace(filename_temp, filename)

@gin.configurable
class ContinuousRunner(base_run_experiment.Runner):
  """Object that handles running Dopamine experiments.

  This is mostly the same as discrete_domains.Runner, but is written solely for
  JAX/Flax agents.
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=gym_lib.create_gym_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=1000,
               checkpoint_every=1,
               clip_rewards=False):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as argument an environment, and
        returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    tf.compat.v1.disable_v2_behavior()
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)
    self._environment = create_environment_fn()

    # import ipdb; ipdb.set_trace()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.
    self._sess = tf.compat.v1.Session('', config=config)
    self._agent = create_agent_fn(self._environment,
                                  summary_writer=self._summary_writer,
                                  sess=self._sess)
    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.compat.v1.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    assert checkpoint_every <= 1, f"deleting checkpoints doesn't work with checkpoint_every > 1, so disabling it. Got {checkpoint_every}"
    # Not ginnable in our version
    self._checkpointer._checkpoint_frequency = checkpoint_every
    print("Checkpointing every", self._checkpointer._checkpoint_frequency)

  def _checkpoint_experiment(self, iteration):
    if self._checkpointer._checkpoint_frequency <= 0:
      # Allow us to skip
      return
    super()._checkpoint_experiment(iteration)

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    metrics = [('Train/NumEpisodes', num_episodes_train),
               ('Train/AverageReturns', average_reward_train),
               ('Train/AverageStepsPerSecond', average_steps_per_second),
               ('Eval/NumEpisodes', num_episodes_eval),
               ('Eval/AverageReturns', average_reward_eval)]
    for name, value in metrics:
      self._summary_writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)]), iteration)



@gin.configurable
class CCExplorationRunner(ContinuousRunner):
    # https://github.com/google/dopamine/blob/202fa9e90aa61edabca92bc4b6f4f7895b95293b/dopamine/discrete_domains/run_experiment.py
    def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn,
               log_trajectories=False,
               skip_bonus_logging=False,
               min_steps_per_second=-1,
               *args, **kwargs):
        # We need some of these things in super if we're loading checkpoints
        self.iteration_number = 0 # why am I doing this part too? This seems bad. Actually, seems like its just not used by them.
        self.count_dir = os.path.join(base_dir, "counts")
        os.makedirs(self.count_dir, exist_ok=True)
        # State counts: state -> count
        self.state_counts = defaultdict(int)
        # episode -> true/approx -> state -> count
        self.count_logs = defaultdict(dict)

        super(CCExplorationRunner, self).__init__(
            base_dir=base_dir,
            create_agent_fn=create_agent_fn,
            create_environment_fn=create_environment_fn,
            *args,
            **kwargs
        )

        self.log_trajectories = log_trajectories
        if self.log_trajectories:
            self.trajectory_dir = os.path.join(base_dir, "trajectories")
            os.makedirs(self.trajectory_dir, exist_ok=True)
            self.is_first_ep_of_iter = True
            self.stored_frames = []
            self.stored_intrinsic_rewards = []

        self.skip_bonus_logging = skip_bonus_logging

        self.min_steps_per_second = min_steps_per_second

    def state2obs(self, state):
        raise NotImplementedError()
    
    def _run_one_step(self, action):
        # Note: I think this adds the "terminal" observation, that only appears
        # in s', which ours by design doesn't.
        # Not a huge deal, it's just logging.
        observation, reward, is_terminal = super()._run_one_step(action)
        info = self._environment.get_current_info()

        if not self._agent.eval_mode:
            self.log(observation, reward, is_terminal, info)

        return observation, reward, is_terminal

    def get_key_from_info(self, info):
        raise NotImplementedError()

    def log(self, obs, reward, is_terminal, info):
        # print('are we even logging')
        if not self.skip_bonus_logging: # This can be slow, and we often don't care.
            key = self.get_key_from_info(info)
            self.state_counts[key] += 1

    # def get_obs_value(self, obs):
    #     # Add batch and stack
    #     obs = obs[None, ...]
    #     obs = obs[..., None]
    #     agent = self._agent
    #     state_qvalues = agent._sess.run(agent._net_outputs.q_values, feed_dict={agent.state_ph: obs})
    #     state_value = state_qvalues.max()
    #     return state_value

    def get_intrinsic_reward(self, obs):
        if isinstance(self._agent, BaseIntrinsicSACAgent):
            rf = self._agent.intrinsic_model.compute_intrinsic_reward
            scaled_intrinsic_reward = rf(obs, self._agent.training_steps, eval_mode=True)
            scale = self._agent.intrinsic_model.reward_scale
            assert np.isscalar(scale), scale
            if scale > 0:
                return scaled_intrinsic_reward / scale
        return 0.

    def log_custom_quantities(self):
        intrinsic_reward_dict = dict()
        # value_dict = dict()
        for state in self.state_counts:
            obs = self.state2obs(state)
            r_int = self.get_intrinsic_reward(obs)
            intrinsic_reward_dict[state] = r_int
            # value_dict[state] = self.get_obs_value(obs)

        self.count_logs[self.iteration_number]["true"] = deepcopy(self.state_counts)
        self.count_logs[self.iteration_number]["approx"] = deepcopy(intrinsic_reward_dict)
        # self.count_logs[self.iteration_number]["value"] = deepcopy(value_dict)

    def _run_one_iteration(self, iteration):
        self.iteration_number = iteration
        # Set to true here
        self.is_first_ep_of_iter = True
        x = super()._run_one_iteration(iteration)
        self.log_custom_quantities()
        self.write_dict_to_file(self.count_logs)
        return x

    @gin.configurable
    def write_dict_to_file(self, dictionary, skip=False):
        if skip:
            return
        t0 = time.time()
        filename = f"{self.count_dir}/count_dict.pkl.gz"
        _safe_zip_write(filename, dictionary)
        print(f"Took {time.time() - t0}s to write count dict to file")


class AntMazeExplorationRunner(CCExplorationRunner):
    """Exploration runner for AntMaze."""
    def __init__(self,
                base_dir,
                create_agent_fn,
                create_environment_fn,
                *args,
                **kwargs):
        super().__init__(base_dir, create_agent_fn, create_environment_fn, *args, **kwargs)

        # map state -> image
        self.state_to_obs = dict()

    # def state2obs(self, state):
    #     position = np.round(state[:2], decimals=1)
    #     return position[0], position[1]
    def state2obs(self, state):
        return self.state_to_obs[state]

    def get_key_from_info(self, info):
        pos = info['player_pos']
        # bucket in 0.2 increments
        rounded_pos = 0.2 * (np.array(pos) // 0.2)
        return tuple(rounded_pos)
    
    @gin.configurable
    def log_custom_quantities(self, skip=False):
        if skip:
            return
        t0 = time.time()
        super().log_custom_quantities()
        print(f"Took {time.time() - t0}s to log custom quantities")
    
    def log(self, obs, reward, is_terminal, info):
        if not self.skip_bonus_logging: # This can be slow, and we often don't care.
            dict_key = self.get_key_from_info(info)
            if dict_key not in self.state_to_obs:
                self.state_to_obs[dict_key] = obs.squeeze()
        return super().log(obs, reward, is_terminal, info)


class HumanoidExplorationRunner(AntMazeExplorationRunner):
  def get_key_from_info(self, info):
    pos = info['player_pos']
    # bucket in 0.01 increments
    rounded_pos = 0.01 * (np.array(pos) // 0.01)
    return tuple(rounded_pos)

class DoorExplorationRunner(AntMazeExplorationRunner):
  def get_key_from_info(self, info):
    pos = info['player_pos']
    return (0, 0) # door isnt exploding but removed just to be safe

class PenExplorationRunner(AntMazeExplorationRunner):
  # For some of the tasks, we need coarser discretization
  def get_key_from_info(self, info):
    pos = info['player_pos']
    return (0, 0) # disabled for now because the CDs are growing out of control

class FetchExplorationRunner(AntMazeExplorationRunner):
  # For some of the tasks, we need coarser discretization
  def get_key_from_info(self, info):
    # pos = info['fetch_pos']
    return (0, 0) # disabled for now because the CDs are growing out of control
