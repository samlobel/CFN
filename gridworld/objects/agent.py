import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy

class Agent(BaseSprite):
    def __init__(self, position=(0, 0), name='Alice'):
        self.position = np.asarray(position)

    def plot(self, ax, linewidth_multiplier=1.0):
        xy = pos2xy(self.position) + (0.5, 0.5)
        c = plt.Circle(xy, 0.3, color='k', fill=False, linewidth=1 * linewidth_multiplier)
        ax.add_patch(c)
