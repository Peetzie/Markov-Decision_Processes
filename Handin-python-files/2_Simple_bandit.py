import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import seaborn as sns
import pandas as pd
from OSHelper import GENV

sns.set_style("dark")


class Bandit:
    def __init__(self, k_arm=10, epsilon=0.0, initial=0.0, sample_averages=False):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        # self.sample_averages = sample_averages
        # self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

        self.average_reward = 0

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        # Use normal distribution for accurate true data
        self.q_true = np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        return np.argmax(self.q_estimation)

    def step(self, action):
        # Generate reward
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1
        self.q_estimation[action] += (
            reward - self.q_estimation[action]
        ) / self.action_count[action]
        self.average_reward += (reward - self.average_reward) / self.action_count

        return reward


def plot_2_1(k):
    # We add k to the array to introduce variability for each of the rewards
    dataset = np.random.randn(1000, k) + np.random.randn(k)
    fig = plt.figure(figsize=(10, 5))
    sns.violinplot(data=dataset)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    env.save_img(img=fig, name="Reward_disribution")


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_2_2(runs=8000, time=2000):
    epsilons = [0, 0.1, 0.01, 1]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    fig = plt.figure(figsize=(10, 5))
    for eps, rewards in zip(epsilons, rewards):
        sns.lineplot(rewards, label="$\epsilon = %.02f$" % (eps))
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.legend()
    plt.title("Average reward (mean) over simulation steps")
    env.save_img(img=fig, name="Mean_Reward")

    fig = plt.figure(figsize=(10, 5))
    for eps, counts in zip(epsilons, best_action_counts):
        sns.lineplot(counts, label="$\epsilon = %.02f$" % (eps))
    plt.xlabel("steps")
    plt.ylabel("% optimal action")
    plt.legend()
    plt.title("Optimal action (mean) over simulation steps")
    env.save_img(img=fig, name="optimal_actions")


if __name__ == "__main__":
    env = GENV(chap=2)
    env.createResDir()
    env.createResDir(images=True)
    plot_2_1(k=10)
    figure_2_2()
