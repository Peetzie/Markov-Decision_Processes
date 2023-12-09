import numpy as np
from OSHelper import GENV


class SimpleGridWorld:
    """
    Example Gridworld from Barton and Sutton - Introduction to Reinforcement Learning
    """

    def __init__(self) -> None:
        self.WORLD_SIZE = 4
        self.DISCOUNT = 0.9

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

    def iterative_policy_evaluation(self):
        """
        Evaluates a random policy without optimizing it as the optimal value is never selected.
        """
        v = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        delta = 1e-4
        while True:
            # Loop for each s in S
            V = np.zeros_like(v)
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    # Loop over actions
                    for action in self.ACTIONS:
                        (nextI, nextJ), reward = self.step([i, j], action)
                        # Bellmann
                        V[i, j] += self.ACTION_PROB * (
                            reward + self.DISCOUNT * v[nextI, nextJ]
                        )
            if np.sum(np.abs(v - V)) < delta:
                break
            v = V
        return v.round(1)

    def optimal_policy_evaluation_w_convergence_steps(self):
        """
        Evaluates the optimal policy by selecting the action that maximizes the reward. There for returning optimal value function,
        It saves each v in an array and returns that.
        """
        v = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        vk = []
        delta = 1e-4
        while True:
            V = np.zeros_like(v)
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    values = []
                    for action in self.ACTIONS:
                        (nextI, nextJ), reward = self.step([i, j], action=action)
                        # expected reward
                        val = reward + self.DISCOUNT * v[nextI, nextJ]
                        values.append(val)
                    # Select value that optimizes the reward of all actions.
                    V[i, j] = np.max(values)
            if np.sum(np.abs(v - V)) < delta:
                break
            v = V
            vk.append(v.round(1))  # Append for convergence array
        return vk

    def optimal_policy_policy_evaluation(self):
        """
        Evaluates the optimal policy by selecting the action that maximizes the reward. There for returning optimal value function
        """
        v = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        delta = 1e-4
        while True:
            V = np.zeros_like(v)
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    values = []
                    for action in self.ACTIONS:
                        (nextI, nextJ), reward = self.step([i, j], action=action)
                        # expected reward
                        val = reward + self.DISCOUNT * v[nextI, nextJ]
                        values.append(val)
                    # Select value that optimizes the reward of all actions.
                    V[i, j] = np.max(values)
            if np.sum(np.abs(v - V)) < delta:
                break
            v = V
        return v.round(1)


if __name__ == "__main__":
    env = GENV(chap=4)
    env.createResDir()
    gw = SimpleGridWorld()
    # Perform value iteration

    value_function = gw.iterative_policy_evaluation()
    env.save_value_iter(value_function, "iterative_policy_evaluation")

    # find optimal value function
    optimal_value_function = gw.optimal_policy_policy_evaluation()
    env.save_value_iter(
        value_function=optimal_value_function, assignment="Optimal_value_function"
    )

    # Convergence steps
    conv_steps = gw.optimal_policy_evaluation_w_convergence_steps()
    env.save_value_iter(
        value_function=conv_steps,
        assignment="Optimal_value_function_convergence_steps",
        with_steps=True,
    )
