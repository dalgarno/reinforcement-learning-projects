import numpy as np
from tile_coding import tiles, IHT
from agent import Agent


class NStepSARSAAgent(Agent):
    def __init__(self, env, weights, num_tilings=8,
                 alpha=0.01, eps=0.2, gamma=0.98):
        super().__init__(env, alpha, eps, gamma)
        self.weights = weights
        self.iht = IHT(len(weights))
        self.num_tilings = num_tilings

    def active_tiles(self, s, a):
        x, v, theta, theta_dot = s

        #  These bounds are required for the tile coding to condense the
        #  continuous space, rather than extend the features out to
        #  infinity
        (x_max, v_max, theta_max,
            theta_dot_max) = self.env.observation_space.high

        (x_min, v_min, theta_min,
            theta_dot_min) = self.env.observation_space.low

        x_bound = self.num_tilings * x / (x_max - x_min)
        v_bound = self.num_tilings * v / (v_max - v_min)
        theta_bound = (
            self.num_tilings * theta / (theta_max - theta_min)
        )
        theta_dot_bound = (
            self.num_tilings * theta_dot / (theta_dot_max - theta_dot_min)
        )

        active_tiles = tiles(
            self.iht,
            self.num_tilings,
            [x_bound, v_bound, theta_bound, theta_dot_bound],
            [a]
        )

        return active_tiles

    def state_action_values(self, s, a):
        return np.sum(self.weights[self.active_tiles(s, a)])
