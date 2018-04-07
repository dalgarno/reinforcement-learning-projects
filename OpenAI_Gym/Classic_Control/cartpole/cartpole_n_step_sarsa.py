#!/usr/bin/env python
import random
import numpy as np
import gym
from n_step_sarsa_agent import NStepSARSAAgent
from operator import add
import matplotlib as mpl
mpl.use('TkAgg')  # required on macOS
from matplotlib import pyplot as plt


def main():
    weights = np.zeros(4096)
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 1000

    #  for plotting
    num_trials = 50
    num_episodes = 200
    average_rewards = []
    n_vals = [1, 8, 16]
    for n in n_vals:
        total_rewards = [0] * num_episodes
        for i in range(num_trials):
            agent = NStepSARSAAgent(env, weights=weights)
            total_rewards = list(map(
                add, total_rewards, agent.train(num_episodes, n))
                )
        average_rewards.append([x/num_trials for x in total_rewards])

    xs = list(range(num_episodes))
    for i in range(len(n_vals)):
        plt.plot(xs, average_rewards[i], label='n = {}'.format(n_vals[i]))
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.title('Effect of changing n on average reward')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    random.seed(101)
    main()
