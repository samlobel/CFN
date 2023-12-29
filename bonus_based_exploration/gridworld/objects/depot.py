import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy
from .. import utils

class Depot(BaseSprite):
    def __init__(self, position=(0, 0), color='red'):
        self.position = np.asarray(position)
        self.color = color

    def plot(self, ax, linewidth_multiplier=1.0):
        xy = pos2xy(self.position) + (0.1, 0.1)
        colorname = utils.get_good_color(self.color)
        c = plt.Rectangle(xy,
                          0.8,
                          0.8,
                          color=colorname,
                          fill=False,
                          linewidth=1.0 * linewidth_multiplier)
        ax.add_patch(c)
