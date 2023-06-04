import gym
import gin
from d4rl.utils.quatmath import quat2euler, euler2quat
import numpy as np

_STAND_HEIGHT = 1.0

class RenderWrapper(gym.Wrapper):
  def step(self, *args, **kwargs):
    self.render(mode="human")
    return super().step(*args, **kwargs)


def _unwrap(env):
  if hasattr(env, 'env'):
    return _unwrap(env.env)
  elif hasattr(env, 'environment'):
    return env.environment
  else:
    return env

class ConditionalRenderWrapper(gym.Wrapper):

  def __init__(self, env, filename):
    self.filename = filename
    self.render_option = self._get_render_option()
    super().__init__(env)

  def _get_render_option(self):
    try:
      with open(self.filename, 'r') as f:
        content = f.read().strip()
      if content == "1":
        return True
      elif content == "0":
        return False
      else:
        raise Exception(f"Not a 0 or 1 in file, instead got {content}")
    except Exception as e:
      print("Got exception:", e)
      print("Not rendering by default.")
      return False

  def _close_viewer(self):
    env = _unwrap(self)
    if hasattr(env, 'viewer'):
      env.viewer.close()
      # Important, so that gym makes a new viewer automatically
      env.viewer = None


  def reset(self):
    old_render_option = self.render_option
    self.render_option = self._get_render_option()    
    if old_render_option and not self.render_option:
      self._close_viewer()

    if self.render_option:
      self.env.render(mode="human")
    return super().reset()
  
  def step(self, *args, **kwargs):
    if self.render_option:
      self.env.render(mode="human")
    return super().step(*args, **kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
  def __init__(self, env):
    self.num_player_lives = None

    gym.Wrapper.__init__(self, env)

  def reset(self, **kwargs):
    s0 = self.env.reset(**kwargs)
    self.num_player_lives = self.get_num_lives(self.get_current_ram())
    return s0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    info = self.get_current_info(info=info)
    self.num_player_lives = info["lives"]
    return obs, reward, done, info

  def get_current_info(self, info={}):
    ram = self.get_current_ram()

    info["lives"] = self.get_num_lives(ram)
    info["falling"] = self.get_is_falling(ram)
    info["player_x"] = self.get_player_x(ram)
    info["player_y"] = self.get_player_y(ram)
    info["player_pos"] = self.get_current_position()
    info["dead"] = int(info["lives"] < self.num_player_lives)
    info["room"] = self.get_current_room()

    return info

  def get_current_room(self):
    ram = self.get_current_ram()
    assert len(ram) == 128
    room_address_id = 3 # https://github.com/oxwhirl/opiq/blob/b7df5134d8b265972ad70f7ba92b385f1e3fe5f2/src/envs/OpenAI_AtariWrapper.py#L657
    return int(ram[room_address_id])

  def get_current_position(self):
    ram = self.get_current_ram()
    return self.get_player_x(ram), self.get_player_y(ram)

  def get_player_x(self, ram):
    return int(self.getByte(ram, 'aa'))

  def get_player_y(self, ram):
    return int(self.getByte(ram, 'ab'))

  def get_num_lives(self, ram):
    return int(self.getByte(ram, 'ba'))

  def get_is_falling(self, ram):
    return int(int(self.getByte(ram, 'd8')) != 0)

  def get_current_ale(self):
    return _unwrap(self.env).ale

  def get_current_ram(self):
    return self.get_current_ale().getRAM()

  @staticmethod
  def _getIndex(address):
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row*16+col

  @staticmethod
  def getByte(ram, address):
    # Return the byte at the specified emulator RAM location
    idx = MontezumaInfoWrapper._getIndex(address)
    return ram[idx]


class AntMazeEnvWrapper(gym.Wrapper):
  def get_current_info(self, info=None):
    if info is None:
      info = {}
    info["player_pos"] = self.current_xy
    return info

  def reset(self):
    start_state = super().reset()
    self._set_current_xy(start_state)
    return start_state

  def step(self, action):
    next_obs, reward, done, info = super().step(action)
    self._set_current_xy(next_obs)
    info = self.get_current_info(info)
    return next_obs, reward, done, info

  def _set_current_xy(self, state):
    self.current_xy = tuple(state[:2].tolist())


@gin.configurable
class HumanoidStandupEnvWrapper(gym.Wrapper):
  """Env wrapper for humanoid environment. """
  def __init__(self, env, stand_height=_STAND_HEIGHT):
    super().__init__(env)
    self.stand_height = stand_height
    self.game_over = False

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    info["player_pos"] = self.current_z
    return info

  def reset(self):
    start_state = super().reset()
    self.game_over = False
    self._set_current_z(start_state)
    return start_state

  def step(self, action):
    next_obs, _, done, info = super().step(action)
    self._set_current_z(next_obs)
    info = self.get_current_info(info)
    reward = self._sparse_reward_function(next_obs)
    terminal = done or (reward == 1)
    if terminal:
      self.game_over = True
    return next_obs, reward, terminal, info

  def _sparse_reward_function(self, next_obs):
    """Determined the stand height by playing around in gym vis."""
    return float(next_obs[0] > self.stand_height)

  def _set_current_z(self, state):
    """Current z is the height, 0 is for backward compatibility."""
    self.current_z = state[0], 0


@gin.configurable
class HandManipulationEnvWrapper(gym.Wrapper):
  """Wrapper around mujoco/d4rl hand manipulation suite. """
  def __init__(self, env):
    super().__init__(env)
    self.game_over = False

  def get_current_info(self, info=None):
    raise NotImplementedError()

  def reset(self):
    start_state = super().reset()
    self.game_over = False
    return start_state

  def step(self, action):
    next_obs, _, done, info = super().step(action)
    info = self.get_current_info(info)
    reward = float(info['goal_achieved'])
    terminal = done or (reward == 1)
    if terminal:
      self.game_over = True
    return next_obs, reward, terminal, info

  def _get_low(self, low, high):
    return low

  def _get_high(self, low, high):
    return high

  def _get_mid(self, low, high):
    return (low + high) / 2


@gin.configurable
class DoorEnvWrapper(HandManipulationEnvWrapper):
  """Env wrapper for door environment. """
  def __init__(self, env, mode=-1):
    super().__init__(env)
    self.mode = mode

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    door_pos = self.env.environment.get_env_state()['door_body_pos']
    info["player_pos"] = door_pos
    return info

  def reset(self):
    obs = super().reset()
    if self.mode == -1:
      return obs
    if self.mode == 0:
      return self._easy_mode_reset()
    if self.mode == 1:
      return self._medium_mode_reset()
    return self._hard_mode_reset()

  def _easy_mode_reset(self):
    env = self.env.environment
    env.model.body_pos[env.door_bid, 0] = self._get_mid(low=-0.3, high=-0.2)
    env.model.body_pos[env.door_bid,1] = self._get_low(low=0.25, high=0.35)
    env.model.body_pos[env.door_bid,2] = self._get_high(low=0.252, high=0.35)

    env.sim.forward()
    return env.get_obs()

  def _medium_mode_reset(self):
    env = self.env.environment
    env.model.body_pos[env.door_bid, 0] = self._get_mid(low=-0.3, high=-0.2)
    env.model.body_pos[env.door_bid,1] = self._get_mid(low=0.25, high=0.35)
    env.model.body_pos[env.door_bid,2] = self._get_high(low=0.252, high=0.35)

    env.sim.forward()
    return env.get_obs()

  def _hard_mode_reset(self):
    env = self.env.environment
    env.model.body_pos[env.door_bid, 0] = self._get_mid(low=-0.3, high=-0.2)
    env.model.body_pos[env.door_bid,1] = self._get_high(low=0.25, high=0.35)
    env.model.body_pos[env.door_bid,2] = self._get_high(low=0.252, high=0.35)

    env.sim.forward()
    return env.get_obs()

@gin.configurable
class HammerEnvWrapper(HandManipulationEnvWrapper):
  """Env wrapper for door environment. """
  def __init__(self, env, mode=-1):
    super().__init__(env)
    self.mode = mode

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    board_pos = self.env.environment.get_env_state()['board_pos']
    info["player_pos"] = board_pos
    return info

  def reset(self):
    obs = super().reset()
    if self.mode == -1:
      return obs
    if self.mode == 0:
      return self._easy_mode_reset()
    if self.mode == 1:
      return self._medium_mode_reset()
    return self._hard_mode_reset()

  def _easy_mode_reset(self):
    env = self.env.environment
    target_bid = env.model.body_name2id('nail_board')
    env.model.body_pos[target_bid,2] = self._get_low(low=0.1, high=0.25)

    env.sim.forward()
    return env.get_obs()

  def _medium_mode_reset(self):
    env = self.env.environment
    target_bid = env.model.body_name2id('nail_board')
    env.model.body_pos[target_bid,2] = self._get_mid(low=0.1, high=0.25)

    env.sim.forward()
    return env.get_obs()

  def _hard_mode_reset(self):
    env = self.env.environment
    target_bid = env.model.body_name2id('nail_board')
    env.model.body_pos[target_bid,2] = self._get_high(low=0.1, high=0.25)

    env.sim.forward()
    return env.get_obs()

@gin.configurable
class RelocateEnvWrapper(HandManipulationEnvWrapper):
  """Environment wrapper for hand manipulation relocate."""

  def __init__(self, env, mode, reset_off_table=True):
    assert mode in (-1, 0, 1, 2), 'default(-1), easy (0), med (1), hard (2)'
    super().__init__(env)
    assert self.env == env
    assert self.game_over == False
    self.mode = mode
    self.reset_off_table = reset_off_table

  def reset(self):
    obs = super().reset()
    if self.mode == -1:
      return obs
    if self.mode == 0:
      return self._easy_mode_reset()
    if self.mode == 1:
      return self._medium_mode_reset()
    return self._hard_mode_reset()

  def _ball_off_table(self):
    env = self.env.environment

    ball_x, ball_y = env.data.body_xpos[env.obj_bid].ravel()[0:2]

    if ball_x < -0.45 or ball_x > 0.45:
      return True
    if ball_y < -0.45 or ball_y > 0.45:
      return True
    return False

  def _easy_mode_reset(self):
    env = self.env.environment

    env.model.body_pos[env.obj_bid,0] = self._get_mid(low=-0.15, high=0.15)
    env.model.body_pos[env.obj_bid,1] = self._get_mid(low=-0.15, high=0.3)
    env.model.site_pos[env.target_obj_sid, 0] = self._get_mid(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,1] = self._get_mid(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,2] = self._get_low(low=0.15, high=0.35)

    env.sim.forward()
    return env.get_obs()

  def _medium_mode_reset(self):
    env = self.env.environment

    env.model.body_pos[env.obj_bid,0] = self._get_low(low=-0.15, high=0.15)
    env.model.body_pos[env.obj_bid,1] = self._get_low(low=-0.15, high=0.3)
    env.model.site_pos[env.target_obj_sid, 0] = self._get_mid(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,1] = self._get_mid(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,2] = self._get_mid(low=0.15, high=0.35)

    env.sim.forward()
    return env.get_obs()

  def _hard_mode_reset(self):
    env = self.env.environment

    env.model.body_pos[env.obj_bid,0] = self._get_low(low=-0.15, high=0.15)
    env.model.body_pos[env.obj_bid,1] = self._get_low(low=-0.15, high=0.3)
    env.model.site_pos[env.target_obj_sid, 0] = self._get_high(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,1] = self._get_high(low=-0.2, high=0.2)
    env.model.site_pos[env.target_obj_sid,2] = self._get_high(low=0.15, high=0.35)

    env.sim.forward()
    return env.get_obs()

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    object_pos = self.env.environment.get_env_state()['obj_pos']
    info["player_pos"] = object_pos
    return info

  def step(self, action):
    obs, r, done, info = super().step(action)
    if self.reset_off_table and self._ball_off_table():
      done = True
      self.game_over = True
    return obs, r, done, info


@gin.configurable
class PenEnvWrapper(HandManipulationEnvWrapper):
  def __init__(self, env, mode=-1):
    assert mode in (-1, 0, 1, 2), 'default(-1), easy (0), med (1), hard (2)'
    super().__init__(env)
    assert self.env == env
    assert self.game_over == False
    self.mode = mode

  def reset(self):
    obs = super().reset()
    assert self.game_over == False
    if self.mode == -1:
      return obs
    if self.mode == 0:
      return self._easy_mode_reset()
    if self.mode == 1:
      return self._medium_mode_reset()
    return self._hard_mode_reset()

  def _reset_to_orient(self, orientation_tuple):
    assert isinstance(orientation_tuple, tuple) and len(orientation_tuple) == 3
    env = self.env.environment
    desired_orien = np.array(orientation_tuple)
    env.model.body_quat[env.target_obj_bid] = euler2quat(desired_orien)
    env.sim.forward()
    new_start_state = env.get_obs()
    return new_start_state

  def _easy_mode_reset(self):
    # Super close to starting point
    return self._reset_to_orient((0., 1., 0.))

  def _medium_mode_reset(self):
    # Roughly 90 degrees away from starting point
    return self._reset_to_orient((-1., 0., 0.))

  def _hard_mode_reset(self):
    # Roughly 180 degrees away from starting point
    return self._reset_to_orient((-1., -1., 0.))

  def step(self, action):
    return super().step(action)

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    pen_pos = self.env.environment.data.body_xpos[self.env.environment.obj_bid].ravel()
    pen_pos = tuple(pen_pos)
    info["player_pos"] = pen_pos
    return info


@gin.configurable
class FetchEnvWrapper(gym.Wrapper):
  """Wrapper around gym-robotics fetch task."""
  def __init__(self, env, mode=-1):
    super().__init__(env)
    self.mode = mode
    self.game_over = False
    self.observation_space = self._get_flat_observation_space()

  def _flatten_obs(self, obs):
    fetch_observation = obs['observation']
    goal_observation = obs['desired_goal']
    return np.concatenate([fetch_observation, goal_observation])

  def _get_flat_observation_space(self):
    current_observation_space = self.observation_space
    observation_space_length = self.observation_space['observation'].shape[0] + self.observation_space['desired_goal'].shape[0]
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(observation_space_length,), dtype=np.float32)
    return observation_space

  def reset(self):
    start_state = super().reset()
    flattened_start_state = self._flatten_obs(start_state)
    self.game_over = False
    if self.mode == -1:
      return flattened_start_state
    if self.mode == 0:
      return self._easy_mode_reset()
    if self.mode == 1:
      return self._medium_mode_reset()
    return self._hard_mode_reset()

  def _wrapper_set_obj_pos(self, offset):
    assert isinstance(offset, tuple) and len(offset) == 2
    env = self.environment.env
    object_xpos = env.initial_gripper_xpos[:2] + np.array(offset)
    object_qpos = env.sim.data.get_joint_qpos("object0:joint")
    assert object_qpos.shape == (7,)
    object_qpos[:2] = object_xpos
    env.sim.data.set_joint_qpos("object0:joint", object_qpos)
    env.sim.forward()

  def _wrapper_set_goal_pos(self, offset):
    assert isinstance(offset, tuple) and len(offset) == 3
    env = self.environment.env
    goal = env.initial_gripper_xpos[:3].copy()
    if env.has_object:
      goal[2] = env.height_offset
      goal += env.target_offset
    goal += np.array(offset)
    env.goal = goal.copy()

  def _deterministic_reset(self, obj_offset, goal_offset):
    env = self.environment.env
    if env.has_object:
      self._wrapper_set_obj_pos(obj_offset)
    self._wrapper_set_goal_pos(goal_offset)

    new_start_state = env._get_obs()
    new_flattened_start_state = self._flatten_obs(new_start_state)

    return new_flattened_start_state

  def _easy_mode_reset(self):
    raise NotImplemented()

  def _medium_mode_reset(self):
    raise NotImplemented()

  def _hard_mode_reset(self):
    raise NotImplemented()

  def step(self, action):
    next_obs, _, done, info = super().step(action)
    flattened_next_obs = self._flatten_obs(next_obs)
    info = self.get_current_info(info)

    reward = float(info["is_success"])
    terminal = done or (reward == 1)
    if terminal:
      self.game_over = True

    return flattened_next_obs, reward, terminal, info

  def get_current_info(self, info=None):
    if info is None:
      info = {}
    info["fetch_pos"] = (0, 0, 0) # placeholder
    return info


class FetchReachEnvWrapper(FetchEnvWrapper):

  def _easy_mode_reset(self):
    return self._deterministic_reset(None, (-0.05, 0.05, 0.025))

  def _medium_mode_reset(self):
    return self._deterministic_reset(None, (-0.1, 0.1, 0.5))

  def _hard_mode_reset(self):
    return self._deterministic_reset(None, (-0.15, 0.15, 0.75))


class FetchPushEnvWrapper(FetchEnvWrapper):

  def _easy_mode_reset(self):
    return self._deterministic_reset((-0.1, 0.1), (-0.05, 0.05, 0.0))

  def _medium_mode_reset(self):
    return self._deterministic_reset((-0.1, 0.1), (0.025, -0.025, 0.0))

  def _hard_mode_reset(self):
    return self._deterministic_reset((-0.1, 0.1), (0.1, -0.1, 0.0))


class FetchSlideEnvWrapper(FetchEnvWrapper):

  def _easy_mode_reset(self):
    return self._deterministic_reset((-0.05, 0.05), (-0.25, -0.1, 0.0))

  def _medium_mode_reset(self):
    return self._deterministic_reset((-0.05, 0.05), (-0., -0.1, 0.0))

  def _hard_mode_reset(self):
    return self._deterministic_reset((-0.05, 0.05), (0.25, 0.1, 0.0))


class FetchPickAndPlaceEnvWrapper(FetchEnvWrapper):

  def _easy_mode_reset(self):
    return self._deterministic_reset((0.05, 0.05), (-0.05, -0.05, 0.1))

  def _medium_mode_reset(self):
    return self._deterministic_reset((0.05, 0.05), (-0.05, -0.05, 0.2))

  def _hard_mode_reset(self):
    return self._deterministic_reset((0.05, 0.05), (-0.05, -0.05, 0.4))
