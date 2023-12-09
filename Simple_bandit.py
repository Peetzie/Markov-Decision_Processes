import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import seaborn as sns
import pandas as pd


class Bandit:
    def __init__(self, k_arm=10, epsilon=0.0, initial=0.0, sample_averages=False):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.sample_averages = sample_averages
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        return np.argmax(self.q_estimation)

    def step(self, action):
        reward = np.random.randn()  # Reward from a normal distribution
        self.action_count[action] += 1
        if self.sample_averages:
            self.q_estimation[action] += (
                reward - self.q_estimation[action]
            ) / self.action_count[action]
        else:
            self.q_estimation[action] += 0.1 * (reward - self.q_estimation[action])
        return reward


def plot_2_1(k):
    # We add k to the array to introduce variability for each of the rewards
    dataset = np.random.randn(1000, k) + np.random.randn(k)
    plt.figure(figsize=(20, 10))
    sns.violinplot(data=dataset)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()


def simulate(runs, time, bandit):
    rewards = np.zeros((runs, time, bandit.k))
    best_action_counts = np.zeros((runs, time))

    for r in trange(runs, desc="Simulating"):
        bandit.reset()

        for t in range(time):
            action = bandit.act()
            reward = bandit.step(action)
            rewards[r, t, :] = bandit.q_estimation
            best_action_counts[r, t] = (
                1 if action == np.argmax(bandit.q_estimation) else 0
            )

    mean_best_action_counts = best_action_counts.mean(axis=0)
    mean_rewards = rewards.mean(axis=0)

    return mean_best_action_counts, mean_rewards


def plot_figure_2_2(bandit, runs=4000, time=2000):
    epsilons = np.arange(0.001, 0.5, 0.003)
    plt.figure(figsize=(10, 8))

    for epsilon in epsilons:
        bandit.reset()
        bandit.epsilon = epsilon
        best_action_counts, _ = simulate(runs, time, bandit)
        plt.plot(best_action_counts, label=f"$\epsilon = {epsilon}$")

    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Figure 2.2: Greedy-$\epsilon$ Action Selection")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    bandit = Bandit(k_arm=10, epsilon=0.1, initial=0.0, sample_averages=True)
    plot_2_1(k=10)
    plot_figure_2_2(bandit)
