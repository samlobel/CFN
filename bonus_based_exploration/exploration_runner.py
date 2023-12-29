from builtins import NotImplemented, NotImplementedError
import os
import gin
import time
import pickle
import gzip
import numpy as np

from copy import deepcopy
from collections import defaultdict
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains.checkpointer import CHECKPOINT_DURATION
from bonus_based_exploration.intrinsic_motivation.base_intrinsic_agent import IntrinsicDQNAgent
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
class ExplorationRunner(Runner):
    # https://github.com/google/dopamine/blob/202fa9e90aa61edabca92bc4b6f4f7895b95293b/dopamine/discrete_domains/run_experiment.py
    def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn,
               log_trajectories=False,
               skip_bonus_logging=False,
               min_steps_per_second=-1,
               checkpoint_every=1,
               *args, **kwargs):
        # We need some of these things in super if we're loading checkpoints
        self.iteration_number = 0 # why am I doing this part too? This seems bad. Actually, seems like its just not used by them.
        self.count_dir = os.path.join(base_dir, "counts")
        os.makedirs(self.count_dir, exist_ok=True)
        # State counts: state -> count
        self.state_counts = defaultdict(int)
        # episode -> true/approx -> state -> count
        self.count_logs = defaultdict(dict)

        super(ExplorationRunner, self).__init__(
            base_dir=base_dir,
            create_agent_fn=create_agent_fn,
            create_environment_fn=create_environment_fn,
            *args,
            **kwargs
        )

        assert checkpoint_every <= 1, f"deleting checkpoints doesn't work with checkpoint_every > 1, so disabling it. Got {checkpoint_every}"
        # Not ginnable in our version
        self._checkpointer._checkpoint_frequency = checkpoint_every

        self.log_trajectories = log_trajectories
        if self.log_trajectories:
            self.trajectory_dir = os.path.join(base_dir, "trajectories")
            os.makedirs(self.trajectory_dir, exist_ok=True)
            self.is_first_ep_of_iter = True
            self.stored_frames = []
            self.stored_intrinsic_rewards = []

        self.skip_bonus_logging = skip_bonus_logging

        self.min_steps_per_second = min_steps_per_second

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        super()._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
        count_dict_filename = f"{self.count_dir}/count_dict.pkl.gz"
        try:
            with gzip.open(count_dict_filename, 'rb') as f:
                current_count_dict = pickle.load(f)
                max_iteration = max(current_count_dict.keys())
                current_state_counts = current_count_dict[max_iteration]["true"]
                self.count_logs = deepcopy(current_count_dict)
                self.state_counts = deepcopy(current_state_counts)
                print("Successfully loaded count dict and state counts from saved files")
        except FileNotFoundError:
            print(f"Could not load {count_dict_filename}")

    def _checkpoint_experiment(self, iteration):
        if self._checkpointer._checkpoint_frequency <= 0:
            # Allow us to skip
            return
        super()._checkpoint_experiment(iteration)
        self._maybe_clean_up_cpprb_buffer(iteration)

    def _maybe_clean_up_cpprb_buffer(self, iteration_number):
        # https://github.com/google/dopamine/blob/202fa9e90aa61edabca92bc4b6f4f7895b95293b/dopamine/discrete_domains/checkpointer.py
        if not hasattr(self._agent, "intrinsic_model") or not hasattr(self._agent.intrinsic_model, "replay_buffer"):
            return

        checkpointer = self._checkpointer
        checkpoint_dir = self._checkpoint_dir
        stale_iteration_number = iteration_number - (checkpointer._checkpoint_frequency * CHECKPOINT_DURATION)

        if stale_iteration_number >= 0:
            intrinsic_buffer_dir = os.path.join(checkpoint_dir, 'intrinsic_buffer')
            intrinsic_buffer_filename = os.path.join(intrinsic_buffer_dir, f"buffer_{stale_iteration_number}.npz")
            try:
                tf.io.gfile.remove(intrinsic_buffer_filename)
            except tf.errors.NotFoundError:
                # Ignore if file not found.
                logging.info('Unable to remove %s', intrinsic_buffer_filename)

    def _run_one_episode(self, *args, **kwargs):
        """This captures both termination in-game and time-based"""
        to_return = super()._run_one_episode(*args, **kwargs)
        if self.log_trajectories and self.is_first_ep_of_iter:
            print('writing frames')
            self._write_frames_and_rewards()
            self.is_first_ep_of_iter = False
            self.stored_frames = []
            self.stored_intrinsic_rewards = []
            print('wrote frames')
        return to_return

    def state2obs(self, state):
        raise NotImplementedError()

    def reward_function(self, observations):
        rewards = []
        for obs in observations:
            rewards.append(self._agent._get_intrinsic_reward(np.array(obs).reshape((84,84))))
        return rewards


    def _get_image(self, observation):
        """Either returns the observation, or uses the ALE to get the real image"""
        raise NotImplementedError()

    def _write_frames_and_rewards(self):
        filename = f"trajectory_iter_{self.iteration_number}.pkl.gz"
        filename = os.path.join(self.trajectory_dir, filename)

        dictionary = {
            'frames': self.stored_frames,
            'intrinsic_rewards': self.stored_intrinsic_rewards,
        }

        _safe_zip_write(filename, dictionary)

    def _run_one_step(self, action):
        # Note: I think this adds the "terminal" observation, that only appears
        # in s', which ours by design doesn't.
        # Not a huge deal, it's just logging.
        observation, reward, is_terminal = super()._run_one_step(action)
        info = self._environment.get_current_info()

        if not self._agent.eval_mode:
            self.log(observation, reward, is_terminal, info)
        
        # is_first_ep_of_iter falsified in _run_one_episode
        if self.log_trajectories and self.is_first_ep_of_iter:
            image = self._get_image(observation)
            intrinsic_reward = self.get_intrinsic_reward(observation.squeeze())
            self.stored_frames.append(image)
            self.stored_intrinsic_rewards.append(intrinsic_reward)

        return observation, reward, is_terminal

    def get_key_from_info(self, info):
        raise NotImplementedError()

    def log(self, obs, reward, is_terminal, info):
        if not self.skip_bonus_logging: # This can be slow, and we often don't care.
            key = self.get_key_from_info(info)
            self.state_counts[key] += 1

    def get_obs_value(self, obs):
        # Add batch and stack
        obs = obs[None, ...]
        obs = obs[..., None]
        agent = self._agent
        state_qvalues = agent._sess.run(agent._net_outputs.q_values, feed_dict={agent.state_ph: obs})
        state_value = state_qvalues.max()
        return state_value

    def get_intrinsic_reward(self, obs):
        if isinstance(self._agent, IntrinsicDQNAgent):
            scale = self._agent.intrinsic_model.reward_scale
            if scale > 0:
                rf = self._agent.intrinsic_model.compute_intrinsic_reward
                scaled_intrinsic_reward = rf(obs, self._agent.training_steps, eval_mode=True)
                assert np.isscalar(scale), scale
                return scaled_intrinsic_reward / scale
        return 0.

    def log_custom_quantities(self):
        intrinsic_reward_dict = dict()
        value_dict = dict()
        for state in self.state_counts:
            obs = self.state2obs(state)
            r_int = self.get_intrinsic_reward(obs)
            intrinsic_reward_dict[state] = r_int
            value_dict[state] = self.get_obs_value(obs)

        self.count_logs[self.iteration_number]["true"] = deepcopy(self.state_counts)
        self.count_logs[self.iteration_number]["approx"] = deepcopy(intrinsic_reward_dict)
        self.count_logs[self.iteration_number]["value"] = deepcopy(value_dict)

    def _run_one_iteration(self, iteration):
        self.iteration_number = iteration
        # Set to true here
        self.is_first_ep_of_iter = True
        x = super()._run_one_iteration(iteration)
        self.log_custom_quantities()
        self.write_dict_to_file(self.count_logs)
        return x

    def plot(self):
        pass

    @gin.configurable
    def write_dict_to_file(self, dictionary, skip=False):
        if skip:
            return
        t0 = time.time()
        filename = f"{self.count_dir}/count_dict.pkl.gz"
        _safe_zip_write(filename, dictionary)
        print(f"Took {time.time() - t0}s to write count dict to file")

    def _run_train_phase(self, statistics):
        """
        NOTE: This isn't perfect: iterations wait to end when episodes do,
        so sometimes there are more frames than expected in an iteration.
        But we'll just set it conservatively. Plus, it's less of a problem
        when iterations are so long.
        """
        start_time = time.time()
        to_return = super()._run_train_phase(statistics)
        time_delta = time.time() - start_time
        steps_per_second = self._training_steps / time_delta
        if steps_per_second < self.min_steps_per_second:
            print("Exiting slow process: steps per second was " +
                   f"{steps_per_second:9.4f}, which is less than {self.min_steps_per_second}")
            raise Exception("Exiting slow process: steps per second was " +
                   f"{steps_per_second:9.4f}, which is less than {self.min_steps_per_second}")
        return to_return


class GridWorldExplorationRunner(ExplorationRunner):

    def state2obs(self, state):
        return self._environment.sensors.observe(np.array(state))

    def _get_image(self, observation):
        """Just returns the obs"""
        return observation

    def get_key_from_info(self, info):
        pos = info['player_pos']
        return tuple(pos)

    @gin.configurable
    def log_custom_quantities(self, skip=False):
        if skip:
            return
        t0 = time.time()
        super().log_custom_quantities()
        print(f"Took {time.time() - t0}s to log custom quantities")

class TaxiExplorationRunner(ExplorationRunner):
    def __init__(self,
                base_dir,
                create_agent_fn,
                create_environment_fn,
                *args,
                **kwargs):
        super().__init__(base_dir, create_agent_fn, create_environment_fn, *args, **kwargs)

        # map state -> image
        self.state_to_obs = dict()

    def state2obs(self, state):
        return self.state_to_obs[state]

    def _get_image(self, observation):
        """Just returns the obs"""
        return observation

    def get_key_from_info(self, info):
        player_pos = info['player_pos']
        goal_pos = info['goal_pos']
        passenger_pos = info['passenger_pos']
        in_taxi = info['in_taxi']
        key = player_pos + goal_pos + passenger_pos + (in_taxi, )
        return key

    def log(self, obs, reward, is_terminal, info):

        if not self.skip_bonus_logging: # This can be slow, and we often don't care.
            dict_key = self.get_key_from_info(info)
            if dict_key not in self.state_to_obs:
                self.state_to_obs[dict_key] = obs.squeeze()
        return super().log(obs, reward, is_terminal, info)

@gin.configurable
class AtariExplorationRunner(ExplorationRunner):
    def __init__(self,
                base_dir,
                create_agent_fn,
                create_environment_fn,
                *args,
                **kwargs):
        super().__init__(base_dir, create_agent_fn, create_environment_fn, *args, **kwargs)

        # map state -> image
        self.state_to_obs = dict()

        # Rooms visited during training
        self.visited_rooms = set()

    def get_key_from_info(self, info):
        pos = info['player_pos']
        room = info["room"]
        return tuple(pos) + (room,)

    def state2obs(self, state):
        return self.state_to_obs[state]

    def _get_image(self, observation):
        """Gets from ALE"""
        return self._environment.get_current_ale().getScreenRGB()

    def log(self, obs, reward, is_terminal, info):
        self.visited_rooms.add(info["room"])

        if not self.skip_bonus_logging: # This can be slow, and we often don't care.
            dict_key = self.get_key_from_info(info)
            if dict_key not in self.state_to_obs:
                self.state_to_obs[dict_key] = obs.squeeze()
        return super().log(obs, reward, is_terminal, info)

    def log_custom_quantities(self):
        t0 = time.time()
        self.count_logs[self.iteration_number]["visited_rooms"] = deepcopy(self.visited_rooms)
        super().log_custom_quantities()
        print(f"Took {time.time() - t0}s to log custom quantities")

    def get_obs_value(self, obs):
        return 0.
