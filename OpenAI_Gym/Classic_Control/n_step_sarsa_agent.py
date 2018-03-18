import numpy as np
from tile_coding import tiles, IHT
from agent import Agent
import random


class NStepSARSAAgent(Agent):
    def __init__(self, env, weights, num_tilings=8,
                 alpha=0.01, eps=0.2, gamma=0.98):
        super().__init__(env, alpha, eps, gamma)
        self.weights = weights
        self.iht = IHT(len(weights))
        self.num_tilings = num_tilings
        print(self.env.observation_space.high)

    def active_tiles(self, s, a):
        x, v, theta, theta_dot = s

        #  These bounds are required for the tile coding to condense the
        #  continuous space, rather than extend the features out to
        #  infinity
        (x_max, v_max, theta_max,
            theta_dot_max) = self.env.observation_space.high

        (x_min, v_min, theta_min,
            theta_dot_min) = self.env.observation_space.low


        #  The domains for the velocity of the cart and the angular 
        #  velocity of the pole is very large
        theta_dot_max /= 1e37
        theta_dot_min /= 1e37
        v_max /= 1e37
        v_min /= 1e37

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

    def select_action(self, s, find_max=False):
        if random.random() < self.eps and not find_max:
            return random.randint(0, 1)
        return np.argmax([
            self.state_action_values(s, a)
            for a in range(self.env.action_space.n)
            ])

    def train(self, num_episodes, n):
        self.num_episodes = num_episodes

        for episode in range(num_episodes):
            print('episode: {}'.format(episode))
            S_list = []
            R_list = []
            A_list = []
            t = 0

            S_list.append(self.env.reset())
            A_list.append(self.select_action(S_list[t]))
            R_list.append(0)

            T = np.inf
            while True:
                if t < T:
                    (s, r, done, info) = self.env.step(A_list[t])
                    S_list.append(s)
                    R_list.append(r)
                    self.env.render()
                    if done:
                        T = t + 1
                    else:
                        A_list.append(self.select_action(S_list[t + 1]))
                tau = t - n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += R_list[i] * (self.gamma ** (i - tau - 1))
                    if (tau + n) < T:
                        G += (self.gamma ** n) * self.state_action_values(
                            S_list[tau + n], A_list[tau + n])
                    self.weights[self.active_tiles(
                        S_list[tau], A_list[tau])] += self.alpha * \
                        (G - self.state_action_values(S_list[tau], A_list[tau]))
                if tau == (T - 1):
                    print('Episode reward: {}'.format(sum(R_list)))
                    break
                t += 1
                
        self.env.close()
