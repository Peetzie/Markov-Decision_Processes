import numpy as np
from OSHelper import GENV
from colorama import init, Fore, Back, Style
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
# Initialize colorama
init(autoreset=True)
# Initialize helper class
env = GENV(chap=4)


class SimpleGridWorld:
    """
    Example Gridworld from Barton and Sutton - Introduction to Reinforcement Learning
    """

    def __init__(self, WORLD_SIZE=10):
        self.WORLD_SIZE = WORLD_SIZE
        self.DISCOUNT = 1
        self.theta = 1e-4

        # left, up, right, down
        self.ACTIONS = [
            np.array([0, -1]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
        ]
        self.ACTIONS_FIGS = ["←", "↑", "→", "↓"]

        self.ACTION_PROB = 0.25

    def check_bounds(self, state):
        x, y = state

        if x < 0 or x >= self.WORLD_SIZE or y < 0 or y >= self.WORLD_SIZE:
            return False
        else:
            return True

    def end_state_reached(self, state):
        x, y = state
        return (x == 0 and y == 0) or (
            x == self.WORLD_SIZE - 1 and y == self.WORLD_SIZE - 1
        )

    def step(self, state, action):
        if self.end_state_reached(state=state):
            reward = 0
            return state, reward
        next_state = (np.array(state) + action).tolist()
        if not self.check_bounds(next_state):
            next_state = state
        reward = -1
        return next_state, reward

    def value_iteration(self):
        value = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        policy = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE, len(self.ACTIONS)))

        while True:
            delta = 0
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    v = value[i, j]
                    values = []
                    for action in self.ACTIONS:
                        (next_i, next_j), reward = self.step([i, j], action)
                        values.append(reward + self.DISCOUNT * value[next_i, next_j])
                    value[i, j] = max(values)
                    delta = max(delta, abs(v - value[i, j]))
            if delta < self.theta:
                break
            # Convert to policy
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    values = []
                    for action in self.ACTIONS:
                        (next_i, next_j), reward = self.step([i, j], action)
                        values.append(reward + self.DISCOUNT * value[next_i, next_j])
                    best_action = np.argmax(values)
                    policy[i, j, :] = np.eye(len(self.ACTIONS))[best_action]
        return value, policy

    def policy_evaluation(self, policy, value, theta=1e-4):
        while True:
            delta = 0
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    v = value[i, j]
                    value_sum = 0
                    for action_index, action_prob in enumerate(policy[i, j]):
                        if action_prob > 0:
                            (next_i, next_j), reward = self.step(
                                [i, j], self.ACTIONS[action_index]
                            )
                            value_sum += action_prob * (
                                reward + self.DISCOUNT * value[next_i, next_j]
                            )
                    value[i, j] = value_sum
                    delta = max(delta, abs(v - value[i, j]))
            if delta < self.theta:
                break
        return value

    def policy_improvement(self, policy, value):
        policy_stable = True
        for i in range(self.WORLD_SIZE):
            for j in range(self.WORLD_SIZE):
                old_action = np.argmax(policy[i, j])
                values = []
                for action in self.ACTIONS:
                    (next_i, next_j), reward = self.step([i, j], action)
                    values.append(reward + self.DISCOUNT * value[next_i, next_j])
                best_action = np.argmax(values)
                policy[i, j] = np.eye(len(self.ACTIONS))[best_action]
                if old_action != best_action:
                    policy_stable = False
        return policy, policy_stable

    def policy_iteration(self):
        policy = np.ones((self.WORLD_SIZE, self.WORLD_SIZE, len(self.ACTIONS))) / len(
            self.ACTIONS
        )
        value = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        while True:
            value = self.policy_evaluation(policy, value, self)
            policy, policy_stable = self.policy_improvement(policy, value)
            if policy_stable:
                break
        return value, policy

    def policy_to_arrows(self, policy):
        grid_arrows = np.empty((self.WORLD_SIZE, self.WORLD_SIZE), dtype=str)
        for i in range(self.WORLD_SIZE):
            for j in range(self.WORLD_SIZE):
                best_action = np.argmax(policy[i, j])
                grid_arrows[i, j] = self.ACTIONS_FIGS[best_action]
        return grid_arrows

    def generate_episode(self):
        state = [
            np.random.choice(range(self.WORLD_SIZE)),
            np.random.choice(range(self.WORLD_SIZE)),
        ]
        episode = []
        while not self.end_state_reached(state):
            action_index = np.random.choice(len(self.ACTIONS))  # Choose an action index
            action = self.ACTIONS[action_index]  # Get the corresponding action
            next_state, reward = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def monte_carlo(self, episodes=10000):
        returns_sum = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        returns_count = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        values = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))

        for _ in tqdm(range(episodes), desc="Monte carlo simuating"):
            episode = self.generate_episode()
            G = 0
            for state, _, reward in reversed(episode):
                G = reward + self.DISCOUNT * G
                returns_sum[state[0], state[1]] += G
                returns_count[state[0], state[1]] += 1
                values[state[0], state[1]] = (
                    returns_sum[state[0], state[1]] / returns_count[state[0], state[1]]
                )

        return values


if __name__ == "__main__":
    w = SimpleGridWorld()
    value, policy = w.value_iteration()
    value_2, policy_2 = w.policy_iteration()
    print(value)
    print(value_2)
    value_3 = w.monte_carlo()
    print(value_3)
