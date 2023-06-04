import gin
import gym
import random
import numpy as np
from gym import spaces
from .gridworld import GridWorld
from .sensors import *


@gin.configurable
class GridWorldEnv(gym.Env):
    def __init__(self,
                 size,
                 randomize_starts,
                 random_goal,
                 noise_scale=0.0, # 0.0 is off, 0.05 seems pretty high.
                 action_noise_prob=0.0,  # stochastic transitions
                 ignore_extrinsic_rewards=False,
                 max_steps_per_episode=1000):
        assert isinstance(randomize_starts, bool), randomize_starts
        assert isinstance(size, int) and size > 0 and 84 % size == 0, size
        rows = cols = size

        # Underlying MDP
        self.gridworld = GridWorld(rows, cols, randomize_starts, random_goal)

        # For measuring pure exploratory behavior
        self.ignore_extrinsic_rewards = ignore_extrinsic_rewards

        # Image dimensions
        img_width = img_height = 84
        assert img_width % rows == 0
        self.scale = img_width // self.gridworld._rows
        
        # Observation params
        self.noise_scale = noise_scale
        self.sensors = self._get_sensors()
        
        # Gym metadata
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # Episode counter
        self.T = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.game_over = False

        self.action_noise_prob = action_noise_prob

        super().__init__()

    def _get_action_space(self):
        n_actions = len(self.gridworld.actions)
        return spaces.Discrete(n_actions)

    def _get_observation_space(self):
        def get_image_sensor():
            for sensor in self.sensors.sensors:
                if isinstance(sensor, ImageSensor):
                    return sensor
            raise NotImplementedError("GymEnv for pure GridWorld")
        return spaces.Box(low=0, high=255, shape=get_image_sensor().size, dtype=np.uint8)

    def _get_sensors(self):
        if self.noise_scale > 0:
            sensor_list = [
                OffsetSensor(offset=(0.5, 0.5)),
                NoisySensor(sigma=0.05),
                ImageSensor(range=((0, self.gridworld._rows),
                                   (0, self.gridworld._cols))),
                ResampleSensor(scale=self.scale),
                BlurSensor(sigma=0.6, truncate=1.),
                NoisySensor(sigma=self.noise_scale),
                MultiplySensor(scale=255),
                ClipSensor(limit_min=0, limit_max=255),
                AsTypeSensor(np.uint8),
            ]
        else:
            sensor_list = [
                OffsetSensor(offset=(0.5, 0.5)),
                ImageSensor(range=((0, self.gridworld._rows),
                                   (0, self.gridworld._cols))),
                ResampleSensor(scale=self.scale),
                MultiplySensor(scale=255),
                AsTypeSensor(np.uint8),
            ]
        return SensorChain(sensor_list)
        
    def step(self, action):
        self.T += 1

        # Stochastic transition function
        if random.random() < self.action_noise_prob:
            action = random.choice(self.gridworld.actions)

        state, reward, done = self.gridworld.step(action)

        if self.ignore_extrinsic_rewards:
            reward = 0.
            done = False

        obs = self.sensors.observe(state)
        reset = self.T % self.max_steps_per_episode == 0
        self.game_over = reset or done
        return obs, reward, done or reset, dict(
            state=state, 
            needs_reset=reset
        )

    def reset(self):
        self.T = 0
        self.gridworld.reset()
        state = self.gridworld.get_state()
        return self.sensors.observe(state)

    def get_current_info(self):
        return dict(
            player_pos=self.gridworld.agent.position,
            goal_pos=self.gridworld.goal.position
        )
    
    def render(self, mode='rgb_array'):
        assert mode in ("human", "rgb_array"), mode

        if mode == "rgb_array":
            return self.sensors.observe(self.gridworld.get_state())
        
        raise NotImplementedError(mode)
        
