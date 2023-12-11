import numpy as np
from OSHelper import GENV
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")


class SimpleGridWorld:
    """
    Example Gridworld from Barton and Sutton - Introduction to Reinforcement Learning
    """

    def __init__(self, WORLD_SIZE=4):
        self.WORLD_SIZE = WORLD_SIZE
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

    def value_iteration(self):
        V = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        Delta = 1e-4

        while True:
            delta = 0
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    old_v = V[i, j]
                    values = []
                    for action in self.ACTIONS:
                        (nextI, nextJ), reward = self.step([i, j], action=action)
                        # Expected reward
                        val = reward + self.DISCOUNT * V[nextI, nextJ]
                        values.append(val)
                    V[i, j] = np.max(values)
                    delta = max(delta, abs(old_v - V[i, j]))
            if delta < Delta:
                break
        return V.round(1)

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

    def linear_solver(self):
        """
        Solve the system of linear equation to find an exact solution.
        """
        A = -1 * np.eye(self.WORLD_SIZE * self.WORLD_SIZE)

        b = np.zeros(self.WORLD_SIZE * self.WORLD_SIZE)
        for i in range(self.WORLD_SIZE):
            for j in range(self.WORLD_SIZE):
                s = [i, j]  # current state
                index_s = np.ravel_multi_index(s, (self.WORLD_SIZE, self.WORLD_SIZE))
                for a in self.ACTIONS:
                    s_, r = self.step(s, a)
                    index_s_ = np.ravel_multi_index(
                        s_, (self.WORLD_SIZE, self.WORLD_SIZE)
                    )

                    A[index_s, index_s_] += self.ACTION_PROB * self.DISCOUNT
                    b[index_s] -= self.ACTION_PROB * r

        x = np.linalg.solve(A, b)
        return x.round(1)

    def timer_comparison(self, num_itrations=500):
        solvers = [
            self.iterative_policy_evaluation,
            self.optimal_policy_policy_evaluation,
            self.optimal_policy_evaluation_w_convergence_steps,
            self.value_iteration,
            self.linear_solver,
        ]
        all_times = []

        for solver_function in solvers:
            times = []
            for i in tqdm(range(num_itrations)):
                start_time = time.time()
                solver_function()
                end_time = time.time()
                times.append(end_time - start_time)

            average_time = sum(times) / num_itrations
            all_times.append(times)
            print(f"{solver_function.__name__} - Average Time: {average_time} seconds")

        fig = plt.figure(figsize=(10, 6))
        for i, solver_function in enumerate(solvers):
            sns.lineplot(all_times[i], label=solver_function.__name__)

        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.title("Solver Comparison")
        env.createResDir(images=True)
        env.save_img(img=fig, name=f"{solver_function.__name__}_{self.WORLD_SIZE}")


if __name__ == "__main__":
    env = GENV(chap=4)
    env.createResDir()
    gw = SimpleGridWorld()
    # Perform value iteration

    start_time = time.time()
    value_function = gw.iterative_policy_evaluation()
    print(value_function)
    end_time = time.time()
    env.save_value_iter(value_function, "iterative_policy_evaluation")
    print(f"Iterative policy evaluation took {end_time-start_time} seconds")

    # find optimal value function
    start_time = time.time()
    optimal_value_function = gw.optimal_policy_policy_evaluation()
    end_time = time.time()
    env.save_value_iter(
        value_function=optimal_value_function, assignment="Optimal_value_function"
    )
    print(f"Policy evlaution took: {end_time-start_time} seconds")

    # Convergence steps
    conv_steps = gw.optimal_policy_evaluation_w_convergence_steps()
    env.save_value_iter(
        value_function=conv_steps,
        assignment="Optimal_value_function_convergence_steps",
        with_steps=True,
    )
    start_time = time.time()
    values = gw.value_iteration()
    end_time = time.time()
    env.save_value_iter(value_function=values, assignment="value_iteration")
    print(f"Value iteration took {end_time-start_time} seconds")
    print(type(values))

    # Linear approach
    start_time = time.time()
    v = gw.linear_solver()
    end_time = time.time()
    v = v.reshape((gw.WORLD_SIZE, gw.WORLD_SIZE))  # Convert to grid to create tabular.
    env.save_value_iter(value_function=v, assignment="Linear Programming approach")
    print(f"The linear approach took {end_time-start_time} seconds")

    # Combined testing method
    gw.timer_comparison(num_itrations=1000)

    # Larger world
    gw = SimpleGridWorld(WORLD_SIZE=30)
    gw.timer_comparison(num_itrations=1000)
