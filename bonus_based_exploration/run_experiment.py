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

"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L
import os

from bonus_based_exploration.intrinsic_motivation import intrinsic_dqn_agent
from bonus_based_exploration.intrinsic_motivation import intrinsic_rainbow_agent
from bonus_based_exploration.noisy_networks import noisy_dqn_agent
from bonus_based_exploration.noisy_networks import noisy_rainbow_agent
from bonus_based_exploration.wrappers import RenderWrapper, ConditionalRenderWrapper, MontezumaInfoWrapper
from bonus_based_exploration.exploration_runner import AtariExplorationRunner, GridWorldExplorationRunner, TaxiExplorationRunner

try:
  from bonus_based_exploration.gridworld.gridenv import GridWorldEnv
except:
  print("Failed to import GridWorldEnv")

try:
  from bonus_based_exploration.taxi.taxi_env import BWTaxiEnv
except:
  print("Failed to import BWTaxiEnv")

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib
from dopamine.agents.dqn.dqn_agent import DQNAgent
import gin
import gym


def create_atari_environment(*args, **kwargs):
  environment = atari_lib.create_atari_environment(*args, **kwargs)
  # assert environment.environment._game == 'MontezumaRevenge', 'Only MontezumaRevenge is supported ... for now!'
  environment = MontezumaInfoWrapper(environment)
  return environment

def create_render_environment_function(*args, **kwargs):
  environment = create_atari_environment(*args, **kwargs)
  environment = RenderWrapper(environment)
  return environment

def create_conditional_render_environment_function(*args, **kwargs):
  environment = create_atari_environment(*args, **kwargs)
  environment = ConditionalRenderWrapper(environment, "bonus_based_exploration/render.txt")
  return environment

@gin.configurable
def gridworld_create_environment_function(*args, **kwargs):
  env = GridWorldEnv()
  # Rows and cols are in `gin config`, as is `max_steps_per_episode`
  return env

@gin.configurable
def taxi_create_environment_function(*args, **kwargs):
  env = BWTaxiEnv()
  # Rows and cols are in `gin config`, as is `max_steps_per_episode`
  return env


@gin.configurable
def create_exploration_agent(sess, environment, agent_name=None,
                             summary_writer=None, debug_mode=False):
  """Creates an exploration agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create. Agent supported are dqn_cts
      and rainbow_cts.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn_pixelcnn':
    return intrinsic_dqn_agent.PixelCNNDQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow_pixelcnn':
    return intrinsic_rainbow_agent.PixelCNNRainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  if agent_name == 'dqn_rnd':
    return intrinsic_dqn_agent.RNDDQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow_rnd':
    return intrinsic_rainbow_agent.RNDRainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  if agent_name == 'noisy_dqn':
    return noisy_dqn_agent.NoisyDQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'noisy_rainbow':
    return noisy_rainbow_agent.NoisyRainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  if agent_name == 'dqn_coinflip':
    return intrinsic_dqn_agent.CoinFlipDQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow_coinflip':
    return intrinsic_rainbow_agent.CoinFlipRainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == "dqn":
    return DQNAgent(
      sess, num_actions=environment.action_space.n,
      summary_writer=summary_writer)
  else:
    return run_experiment.create_agent(sess, environment, agent_name,
                                       summary_writer, debug_mode)


@gin.configurable
def create_exploration_runner(base_dir, create_agent_fn,
                              schedule='continuous_train_and_eval',
                              render=False,
                              conditional_render=False,
                              env_type='atari'):
  """Creates an experiment Runner.

  Args:
    base_dir: Base directory for hosting all subdirectories.
    create_agent_fn: A function that takes as args a Tensorflow session and a
     Gym Atari 2600 environment, and returns an agent.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `run_experiment.Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and eval till max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    if env_type == 'atari':
      if render:
        return AtariExplorationRunner(base_dir, create_agent_fn, create_environment_fn=create_render_environment_function)
      elif conditional_render:
        return AtariExplorationRunner(base_dir, create_agent_fn, create_environment_fn=create_conditional_render_environment_function)
      else:
        return AtariExplorationRunner(base_dir, create_agent_fn, create_environment_fn=create_atari_environment)
    elif env_type == 'gridworld':
      print("Creating gridworld environment..")
      return GridWorldExplorationRunner(base_dir, create_agent_fn, create_environment_fn=gridworld_create_environment_function)
    elif env_type == 'taxi':
      print("Creating taxi environment..")
      return TaxiExplorationRunner(base_dir, create_agent_fn, create_environment_fn=taxi_create_environment_function)
    else:
      raise Exception(f'env_type must be one of atari, gridworld, taxi. Got {env_type}')
  # Continuously runs training till maximum num_iterations is hit.
  elif schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_fn)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))
