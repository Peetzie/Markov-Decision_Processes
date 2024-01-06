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

    def __init__(self, WORLD_SIZE=4):
        self.WORLD_SIZE = WORLD_SIZE
        self.DISCOUNT = 0.99999

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

    def policy_evaluation(self):
        """
        Evaluates and improves the optimal policy by selecting the action that maximizes the reward.
        Returns both the optimal value function and the optimal policy.
        """
        v = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        # Initialize the policy with a default action, e.g., 0 (left)
        policy = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE), dtype=int)
        delta = 1e-4
        policy_stable = False

        while not policy_stable:
            # Policy Evaluation
            while True:
                V = np.zeros_like(v)
                for i in range(self.WORLD_SIZE):
                    for j in range(self.WORLD_SIZE):
                        action = self.ACTIONS[
                            policy[i, j]
                        ]  # Action according to current policy
                        (nextI, nextJ), reward = self.step([i, j], action)
                        V[i, j] = reward + self.DISCOUNT * v[nextI, nextJ]

                if np.sum(np.abs(v - V)) < delta:
                    break
                v = V

            # Policy Improvement
            policy_stable = True
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    old_action = policy[i, j]
                    action_values = []
                    for a_idx, action in enumerate(self.ACTIONS):
                        (nextI, nextJ), reward = self.step([i, j], action)
                        action_value = reward + self.DISCOUNT * v[nextI, nextJ]
                        action_values.append(action_value)

                    best_action = np.argmax(action_values)
                    policy[i, j] = best_action

                    if old_action != best_action:
                        policy_stable = False

        return v.round(1), policy

    def reduced_policy_evaluation(self):
        next_state_values = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        # Initialize the policy with a default action, e.g., 0 (left)
        policy = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE), dtype=int)
        delta = 1e-4
        while True:
            state_values = next_state_values.copy()
            for i in range(self.WORLD_SIZE):
                for j in range(self.WORLD_SIZE):
                    value = 0
                    for action in self.ACTIONS:
                        (nextI, nextJ), reward = self.step([i, j], action)
                        value += self.ACTION_PROB * (
                            reward + self.DISCOUNT * state_values[nextI, nextJ]
                        )
                    next_state_values[i, j] = value
            if np.sum(np.abs(state_values - next_state_values)) < delta:
                break

            return next_state_values.round(1), policy

    def timer_comparison(self, num_itrations=500):
        solvers = [
            self.reduced_policy_evaluation,
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


def TEST():
    print(Fore.RED + "[##### TESTING ENVIROMENT #####]")
    gw = SimpleGridWorld()
    env.createResDir()
    # Perform value iteration

    start_time = time.time()
    value_function = gw.iterative_policy_evaluation()
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

    # Linear approach
    start_time = time.time()
    v = gw.linear_solver()
    end_time = time.time()
    v = v.reshape((gw.WORLD_SIZE, gw.WORLD_SIZE))  # Convert to grid to create tabular.
    env.save_value_iter(value_function=v, assignment="Linear Programming approach")
    print(f"The linear approach took {end_time-start_time} seconds")

    # Slimmed policy evalautaion
    start_time = time.time()
    v, _ = gw.reduced_policy_evaluation()
    end_time = time.time()
    v = v.reshape((gw.WORLD_SIZE, gw.WORLD_SIZE))  # Convert to grid to create tabular.
    env.save_value_iter(value_function=v, assignment="Reduced_policy_evaluation")
    print(f"The linear approach took {end_time-start_time} seconds")


def multipleWorldSim():
    print(Fore.GREEN + "#### MULTIPLE WORLD SIM ####")
    WORLD_SIZES = [4, 10, 30, 60, 150, 300]
    for size in WORLD_SIZES:
        print(Fore.GREEN + f"Simulating world_size: {size}")
        gw = SimpleGridWorld(WORLD_SIZE=size)
        gw.timer_comparison(num_itrations=100)


if __name__ == "__main__":
    TEST()
    multipleWorldSim()
