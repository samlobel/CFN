import numpy as np

from visgrid.gridworld import skills

class DistanceOracle:
    def __init__(self, env):
        self.env = env
        states = np.indices((env._rows, env._cols)).T.reshape(-1, 2)
        for s in states:
            for sp in states:
                # Pre-compute all pairwise distances
                skills.GoToGridPosition(env, s, sp)

    def pairwise_distances(self, indices, s0, s1):
        init_states = s0[indices]
        next_states = s1[indices]

        distances = [
            skills.GoToGridPosition(self.env, s, sp)[1] for s, sp in zip(init_states, next_states)
        ]

        return distances

#%%
if __name__ == '__main__':
    import seeding
    import numpy as np
    import random

    from visgrid.gridworld import GridWorld, MazeWorld, SpiralWorld
    from visgrid.gridworld import grid
    import matplotlib.pyplot as plt

    grid.directions[3]

    seeding.seed(0, np, random)
    env = SpiralWorld(rows=6, cols=6)
    env.plot()

    oracle = DistanceOracle(env)

    distances = [v[-1] for k, v in env.saved_directions.items()]

    plt.hist(distances, bins=36)
    plt.show()
