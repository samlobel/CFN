import numpy as np
import matplotlib.pyplot as plt
import seeding

grid_type = int

# Offsets:
LEFT = np.asarray((0, -1))
RIGHT = np.asarray((0, 1))
UP = np.asarray((-1, 0))
DOWN = np.asarray((1, 0))
directions = {
    0: LEFT,
    1: RIGHT,
    2: UP,
    3: DOWN,
}
actions = {
    tuple(LEFT): 0,
    tuple(RIGHT): 1,
    tuple(UP): 2,
    tuple(DOWN): 3,
}

class BaseGrid:
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols

        self.wall_color = 'C0'

        # Add rows and columns for walls between cells
        self._grid = np.ones([rows * 2 + 1, cols * 2 + 1], dtype=grid_type)
        self._contents = np.empty([rows * 2 + 1, cols * 2 + 1], dtype=np.object)
        self.saved_directions = {}

        # Reset valid positions and walls
        self._grid[1:-1:2, 1:-1] = 0
        self._grid[1:-1, 1:-1:2] = 0

    def get_random_position(self, seed=None):
        if seed is not None:
            seeding.seed(seed, np)
        return np.asarray((np.random.randint(0, self._rows), np.random.randint(0, self._cols)))

    def contents(self, row, col):
        return self._contents[row // 2, col // 2]

    def plot(self, ax=None, draw_bg_grid=True, linewidth_multiplier=1.0):
        scale = 3 / 5
        rowscale = scale * self._rows
        colscale = scale * self._cols
        if ax is None:
            plt.figure(figsize=(colscale, rowscale))
            ax = plt.axes()
        ax.axis('off')
        ax.axis('equal')
        ax.set_xticks([]), ax.set_yticks([])
        ax.invert_yaxis()

        # Draw faint background grid
        row_range = np.linspace(0, self._rows, self._rows + 1)
        col_range = np.linspace(0, self._cols, self._cols + 1)
        if draw_bg_grid:
            for row in range(self._rows):
                ax.vlines(col_range,
                          row,
                          row + 1,
                          colors='lightgray',
                          linewidth=0.5 * linewidth_multiplier)
            for col in range(self._cols):
                ax.hlines(row_range,
                          col,
                          col + 1,
                          colors='lightgray',
                          linewidth=0.5 * linewidth_multiplier)
        else:
            ax.set_xlim([0, self._cols])
            ax.set_ylim([0, self._rows])

        # Get lists of vertical and horizontal wall locations
        v_walls = self._grid[:, ::2][1::2, :]
        h_walls = self._grid[::2, :][:, 1::2].transpose()
        for row in range(self._rows):
            ax.vlines(col_range[v_walls[row] == 1],
                      row,
                      row + 1,
                      colors=self.wall_color,
                      linewidth=1.0 * linewidth_multiplier)
        for col in range(self._cols):
            ax.hlines(row_range[h_walls[col] == 1],
                      col,
                      col + 1,
                      colors=self.wall_color,
                      linewidth=1.0 * linewidth_multiplier)
        return ax

    def has_wall(self, position, offset):
        row, col = position
        d_row, d_col = offset
        wall_row = 2 * row + 1 + d_row
        wall_col = 2 * col + 1 + d_col
        return self._grid[wall_row, wall_col]

    def add_random_walls(self, n_walls=1):
        types = ['vertical']
        for i in range(n_walls):
            type = np.random.choice(types)
            if type == 'horizontal':
                row = 2 + 2 * np.random.choice(self._rows - 1)
                col = 1 + 2 * np.random.choice(self._cols - 1)
            else:
                row = 1 + 2 * np.random.choice(self._rows - 1)
                col = 2 + 2 * np.random.choice(self._cols - 1)
            self._grid[row, col] = 1

    def save(self, filename):
        np.savetxt(filename, self._grid.astype(int), fmt='%1d')

    def load(self, filename):
        grid = np.loadtxt(filename, dtype=grid_type)
        r, c = grid.shape
        self._rows = r // 2
        self._cols = c // 2
        self._grid = grid
        self._contents = np.empty_like(self._grid, dtype=np.object)
