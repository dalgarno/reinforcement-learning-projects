#!/usr/bin/env python
import random
import numpy as np
import gym
from n_step_sarsa_agent import NStepSARSAAgent
import matplotlib as mpl
mpl.use('TkAgg')  # required on macOS


def main():
    weights = np.zeros(4096)
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 1000
    agent = NStepSARSAAgent(env, weights=weights)
    agent.train(500, 5)


if __name__ == "__main__":
    random.seed(101)
    main()
