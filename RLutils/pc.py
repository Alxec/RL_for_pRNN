from minigrid.core.world_object import Wall
from scipy.stats import norm
import numpy as np

class FakePlaceCells(object):

    def __init__(self, env, size=300, sd=3, normalize=True):
        self.env = env.env
        self.size = size
        self.sd = np.tile(sd, (2, size))
        self.means = np.zeros((2, size), dtype=int)
        if normalize:
            self.norm_factor = norm.pdf(0, 0, sd)**2
        else:
            self.norm_factor = 1
        for i in range(size):
            self.means[:,i] = self.check_position()
        
    def check_position(self):

        while True:

            pos = np.random.randint((0,0), (self.env.grid.width, self.env.grid.height))

            # No place fields for wall positions
            if type(self.env.grid.get(*pos)) is Wall:
                continue

            break
        
        return pos

    def activation(self, pos):

        pos = np.tile(np.array(pos), (self.size,1)).T
        probs = norm.pdf(self.means, pos, self.sd)
        return np.multiply(probs[0], probs[1])/self.norm_factor