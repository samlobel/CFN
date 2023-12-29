# coding=utf-8
# Copyright 2021 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from typing import Optional

from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.sac import sac_agent
from flax.metrics import tensorboard
import gin
from gym import spaces
from bonus_based_exploration.cc_intrinsic_motivation.intrinsic_SAC import RNDSACAgent, CoinFlipSACAgent
import tensorflow.compat.v1 as tf
from bonus_based_exploration.cc_intrinsic_motivation.exploration_runner import AntMazeExplorationRunner, HumanoidExplorationRunner
from bonus_based_exploration.cc_intrinsic_motivation.exploration_runner import DoorExplorationRunner, PenExplorationRunner
from bonus_based_exploration.cc_intrinsic_motivation.exploration_runner import FetchExplorationRunner

import d4rl
import gym
import gym_robotics # This has to come after gym!
from bonus_based_exploration.wrappers import (
  AntMazeEnvWrapper, HumanoidStandupEnvWrapper,
  DoorEnvWrapper, PenEnvWrapper, RelocateEnvWrapper, HammerEnvWrapper,
  FetchEnvWrapper, FetchReachEnvWrapper, FetchPushEnvWrapper,
  FetchSlideEnvWrapper, FetchPickAndPlaceEnvWrapper,)

load_gin_configs = base_run_experiment.load_gin_configs

from gym_robotics.envs import FetchReachEnv, FetchPushEnv, FetchSlideEnv, FetchPickAndPlaceEnv

# @gin.configurable
def create_antmaze_environment():
  env = gym_lib.create_gym_environment()
  env = AntMazeEnvWrapper(env)
  return env

# @gin.configurable
def create_humanoid_environment():
  env = gym_lib.create_gym_environment()
  env = HumanoidStandupEnvWrapper(env)
  return env

# @gin.configurable
def create_door_environment():
  env = gym_lib.create_gym_environment()
  env = DoorEnvWrapper(env)
  return env

# @gin.configurable
def create_pen_environment():
  env = gym_lib.create_gym_environment()
  env = PenEnvWrapper(env)
  return env

def create_hammer_environment():
  env = gym_lib.create_gym_environment()
  env = HammerEnvWrapper(env)
  return env


def create_relocate_environment():
  env = gym_lib.create_gym_environment()
  env = RelocateEnvWrapper(env)
  return env

def create_fetch_environment():
  env = gym_lib.create_gym_environment()
  unwrapped_env = env.environment.env
  if isinstance(unwrapped_env, FetchReachEnv):
    return FetchReachEnvWrapper(env)
  if isinstance(unwrapped_env, FetchPushEnv):
    return FetchPushEnvWrapper(env)
  if isinstance(unwrapped_env, FetchSlideEnv):
    return FetchSlideEnvWrapper(env)
  if isinstance(unwrapped_env, FetchPickAndPlaceEnv):
    return FetchPickAndPlaceEnvWrapper(env)
  raise ValueError(f"Should be one of FetchReach, FetchPush, FetchSlide, FetchPickAndPlace. Got {env}")


@gin.configurable
def create_continuous_agent(
    environment: gym_lib.GymPreprocessing,
    agent_name: str,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
    sess=None,
) -> dqn_agent.JaxDQNAgent:
  """Creates an agent.

  Args:
    environment:  A gym environment.
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  assert isinstance(environment.action_space, spaces.Box)
  assert isinstance(environment.observation_space, spaces.Box)
  if agent_name == 'sac':
    return sac_agent.SACAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=None) # SACAgent assumes Flax, which breaks
  elif agent_name == 'sac_rnd':
    return RNDSACAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=summary_writer,
        sess=sess)
  elif agent_name == 'sac_coinflip':
    return CoinFlipSACAgent(
      action_shape=environment.action_space.shape,
      action_limits=(environment.action_space.low,
                      environment.action_space.high),
      observation_shape=environment.observation_space.shape,
      action_dtype=environment.action_space.dtype,
      observation_dtype=environment.observation_space.dtype,
      summary_writer=summary_writer,
      sess=sess
    )
  else:
    raise ValueError(f'Unknown agent: {agent_name}')


@gin.configurable
def create_continuous_exploration_runner(base_dir,
                                         schedule='continuous_train_and_eval',
                                         env_type="ant_maze"):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    if env_type == 'ant_maze':
      return AntMazeExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_antmaze_environment)
    if env_type == 'humanoid':
      return HumanoidExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_humanoid_environment)
    if env_type == 'door':
      return DoorExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_door_environment)
    if env_type == 'pen':
      return PenExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_pen_environment)
    if env_type == 'relocate':
      return PenExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_relocate_environment)
    if env_type == 'hammer':
      return PenExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_hammer_environment)
    if env_type == 'fetch':
      return FetchExplorationRunner(base_dir,
        create_agent_fn=create_continuous_agent,
        create_environment_fn=create_fetch_environment)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))
