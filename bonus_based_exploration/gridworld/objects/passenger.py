import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy
from visgrid import utils

class Passenger(BaseSprite):
    def __init__(self, position=(0, 0), goal='red', color='gray'):
        self.position = np.asarray(position)
        self.goal = goal
        self.color = color
        self.intaxi = False

    def plot(self, ax, linewidth_multiplier=1.0):
        x, y = pos2xy(self.position) + (0.5, 0.5)
        outline_color = utils.get_good_color(self.color)
        fill_color = outline_color  # if self.intaxi else 'white'
        fill_passenger = True  #self.intaxi

        fill = plt.Circle((x, y),
                          0.2,
                          color=fill_color,
                          fill=fill_passenger,
                          linewidth=1 * linewidth_multiplier)
        ax.add_patch(fill)
        outline = plt.Circle((x, y),
                             0.2,
                             color=outline_color,
                             fill=False,
                             linewidth=1 * linewidth_multiplier)
        ax.add_patch(outline)

        if self.intaxi:
            ax.vlines(x, y - 0.2, y + 0.2, colors='black', linewidth=0.5 * linewidth_multiplier)
            ax.hlines(y, x - 0.2, x + 0.2, colors='black', linewidth=0.5 * linewidth_multiplier)
        else:
            ax.plot([x - 0.15, x + 0.15], [y - 0.15, y + 0.15],
                    color='black',
                    markersize=0,
                    linewidth=0.6 * linewidth_multiplier)
            ax.plot([x - 0.15, x + 0.15], [y + 0.15, y - 0.15],
                    color='black',
                    markersize=0,
                    linewidth=0.6 * linewidth_multiplier)
