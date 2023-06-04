from visgrid.envs import TaxiEnv
import numpy as np
from visgrid import utils
from gym import spaces
import gin
import cv2

# from visgrid.wrappers import GrayscaleWrapper # I may not need this.
# I can probably write my own wrapper, or just bake it in. I should just bake it in.

# I think I want to turn off exploring starts.

@gin.configurable
class BWTaxiEnv(TaxiEnv):
    def __init__(self, *args, exploring_starts = False, **kwargs):
        super().__init__(*args, exploring_starts=exploring_starts, **kwargs)
        self.game_over = False

    def reset(self):
        obs, underlying_state = super().reset()
        self.game_over = False
        obs = self._postprocess_obs(obs)
        return obs

    def _initialize_obs_space(self):
        img_shape = self.dimensions['img_shape'] # + (3, )
        # self.img_observation_space = spaces.Box(0.0, 1.0, img_shape, dtype=np.float32)
        self.img_observation_space = spaces.Box(0, 255, img_shape, dtype=np.uint8)

        factor_obs_shape = self.state_space.nvec
        self.factor_observation_space = spaces.MultiDiscrete(factor_obs_shape, dtype=int)

        assert self.should_render is True, self.should_render # Me
        self.set_rendering(self.should_render) # sets observation to img_observation for us.
   
    def _postprocess_obs(self, obs):
        # Greys, subtracts, etc.
        assert obs.shape == (84, 84, 3) or obs.shape == (128, 128, 3)
        obs = obs.mean(axis=2)
        obs = np.where(obs == 1, 0., 1.) # inverts as well so mostly 0s.

        if obs.shape == (128, 128):
            obs = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
            obs = cv2.resize(obs, dsize=(84, 84), interpolation=cv2.INTER_LINEAR)
        obs = obs * 255
        obs = obs.astype(np.uint8)
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if truncated: # Note that previous behavior was for gym 26
            info['TimeLimit.truncated'] = True
        obs = self._postprocess_obs(obs)
        self.game_over = done
        return obs, reward, done, info
    
    def get_current_info(self, info={}):
        # TODO
        assert len(self.passengers) == 1
        passenger = self.passengers[0]
        active_depot = [depot for depot in self.depots.values() if depot.color == passenger.color]
        assert len(active_depot) == 1, active_depot
        active_depot = active_depot[0]

        taxi_row, taxi_col = self.agent.position
        pass_row, pass_col = passenger.position
        goal_row, goal_col = active_depot.position
        in_taxi = passenger.in_taxi

        to_return = dict(
            player_pos=(taxi_row, taxi_col),
            goal_pos=(goal_row, goal_col),
            passenger_pos=(pass_row, pass_col),
            in_taxi=in_taxi
        )

        to_return.update(info) # Note: the other direction would be very wrong.
        return to_return
 
    def _render_objects(self) -> dict:
        walls = self.grid.render(cell_width=self.dimensions['cell_width'],
                                 wall_width=self.dimensions['wall_width'])
        walls = utils.to_rgb(walls, 'almost black')

        depot_patches = np.zeros_like(walls)
        for depot in self.depots.values():
            # Only render depot you're trying to visit
            assert len(self.passengers) == 1
            if depot.color != self.passengers[0].color:
                continue

            patch = self._render_depot_patch(depot.color)
            self._add_patch(depot_patches, patch, depot.position)

        agent_patches = np.zeros_like(walls)
        patch = self._render_character_patch()
        self._add_patch(agent_patches, patch, self.agent.position)

        objects = {
            'walls': walls,
            'depots': depot_patches,
            'agent': agent_patches,
        }
        if self.hidden_goal:
            del objects['depots']

        del objects['agent']

        passenger_patches = np.zeros_like(objects['walls'])
        for p in self.passengers:
            patch = self._render_passenger_patch(p.in_taxi, p.color)
            self._add_patch(passenger_patches, patch, p.position)

        taxi_patches = np.zeros_like(objects['walls'])
        patch = self._render_taxi_patch()
        self._add_patch(taxi_patches, patch, self.agent.position)

        objects.update({
            'taxi': taxi_patches,
            'passengers': passenger_patches,
        })

        return objects

