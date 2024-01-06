import numpy as np
from scipy.stats import poisson
import seaborn as sns
from tqdm import trange
from OSHelper import GENV
import matplotlib.pyplot as plt


class JacksCarRental:
    def __init__(
        self,
        max_cars=20,
        max_move_of_cars=5,
        rental_request_first_loc=3,
        rental_request_second_loc=4,
        returns_first_loc=3,
        returns_second_loc=2,
        discount=0.9,
        rental_credit=10,
        move_car_cost=2,
    ):
        self.MAX_CARS = max_cars
        self.MAX_MOVE_OF_CARS = max_move_of_cars
        self.RENTAL_REQUEST_FIRST_LOC = rental_request_first_loc
        self.RENTAL_REQUEST_SECOND_LOC = rental_request_second_loc
        self.RETURNS_FIRST_LOC = returns_first_loc
        self.RETURNS_SECOND_LOC = returns_second_loc
        self.DISCOUNT = discount
        self.RENTAL_CREDIT = rental_credit
        self.MOVE_CAR_COST = move_car_cost
        self.POISSON_UPPER_BOUND = 11
        self.ACTIONS = np.arange(-self.MAX_MOVE_OF_CARS, self.MAX_MOVE_OF_CARS + 1)
        self.POISSON_CACHE = dict()

    def poisson(self, n, lam):
        key = n * 10 + lam
        # Save values from posson in dict to optimize speed
        if key not in self.POISSON_CACHE:
            self.POISSON_CACHE[key] = poisson.pmf(n, lam)
        return self.POISSON_CACHE[key]

    def get_rewards(self, rental_first, rental_second):
        return (rental_first + rental_second) * self.RENTAL_CREDIT

    def step(self, state, action, v):
        returns = 0.0

        # The cost of moving cars based on action
        returns -= self.MOVE_CAR_COST * abs(action)

        # Moving the cars (Overnight) - state is at the end of a day!
        CARS_FIRST_LOC = min(state[0] - action, self.MAX_CARS)
        CARS_SECOND_LOC = min(state[1] + action, self.MAX_CARS)

        ## NEW DAY!

        # Go through all rental requests and get valid < no. of cars

        for rental_req_first_loc in range(self.POISSON_UPPER_BOUND):
            for rental_req_second_loc in range(self.POISSON_UPPER_BOUND):
                # Probability
                prob = self.poisson(
                    rental_req_first_loc, self.RENTAL_REQUEST_FIRST_LOC
                ) * self.poisson(rental_req_second_loc, self.RENTAL_REQUEST_SECOND_LOC)

                cars_fist_loc = CARS_FIRST_LOC
                cars_second_loc = CARS_SECOND_LOC

                # Valid rentals
                valid_rental_first = min(cars_fist_loc, rental_req_first_loc)
                valid_rental_second = min(cars_second_loc, rental_req_second_loc)

                # Rewards for rentals
                reward = self.get_rewards(valid_rental_first, valid_rental_second)

                # Remove cars from location as they are not rented out.
                cars_fist_loc -= valid_rental_first
                cars_second_loc -= valid_rental_second

                ## Returns
                for returns_first_loc in range(self.POISSON_UPPER_BOUND):
                    for returns_second_loc in range(self.POISSON_UPPER_BOUND):
                        prob_return = self.poisson(
                            returns_first_loc, self.RETURNS_FIRST_LOC
                        ) * self.poisson(returns_second_loc, self.RETURNS_SECOND_LOC)
                        # Compare with rental value
                        cars_first_loc_ = int(
                            min(cars_fist_loc + returns_first_loc, self.MAX_CARS)
                        )
                        cars_second_loc_ = int(
                            min(cars_second_loc + returns_second_loc, self.MAX_CARS)
                        )

                        # Value updating
                        prob_ = prob_return * prob
                        returns += prob_ * (
                            reward
                            + self.DISCOUNT * v[cars_first_loc_, cars_second_loc_]
                        )

        return returns

    def policy_iteration(self):
        """
        Policy iteration to estimate optimal policy - like the book
        """

        value = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        policy = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        kv = []
        kp = []
        delta = 1e-4
        while True:
            while True:
                old_value = value.copy()
                for i in trange(self.MAX_CARS + 1, desc="Policy Evaluation"):
                    for j in range(self.MAX_CARS + 1):
                        new_state_value = self.step([i, j], policy[i, j], value)
                        value[i, j] = new_state_value
                max_value_change = np.abs(old_value - value).max()
                print("max value change {}".format(max_value_change))
                if max_value_change < delta:
                    break
            policy_stable = True
            for i in trange(self.MAX_CARS + 1, desc="Policy evaluation"):
                for j in range(self.MAX_CARS + 1):
                    old_action = policy[i, j]
                    action_returns = []
                    for action in self.ACTIONS:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(self.step([i, j], action, value))
                        else:
                            action_returns.append(-np.inf)
                    new_action = self.ACTIONS[np.argmax(action_returns)]
                    policy[i, j] = new_action
                    if policy_stable and old_action != new_action:
                        policy_stable = False
            kp.append(policy)
            kv.append(value)
            print(f"Policy stable {policy_stable}")
            if policy_stable:
                break
        return kv, kp

    def test(self):
        value = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        policy = np.zeros(value.shape, dtype=int)

        iterations = 0
        _, axes = plt.subplots(2, 3, figsize=(40, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()
        while True:
            fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
            fig.set_ylabel("# cars at first loc", fontsize=30)
            fig.set_yticks(list(reversed(range(self.MAX_CARS + 1))))
            fig.set_xlabel("# cars at sec loc", fontsize=30)
            fig.set_title(f"Policy {iterations}", fontsize=30)

            while True:
                old_value = value.copy()
                for i in trange(
                    self.MAX_CARS + 1, desc=f"Value Iteration, {iterations}"
                ):
                    for j in range(self.MAX_CARS + 1):
                        new_state_value = self.step([i, j], policy[i, j], value)
                        value[i, j] = new_state_value
                max_value_change = np.abs(old_value - value).max()
                print("max value change {}".format(max_value_change))
                if max_value_change < 1e-4:
                    break
            policy_stable = True
            for i in range(self.MAX_CARS + 1):
                for j in range(self.MAX_CARS + 1):
                    old_action = policy[i, j]
                    action_returns = []
                    for action in self.ACTIONS:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(self.step([i, j], action, value))
                        else:
                            action_returns.append(-np.inf)
                    new_action = self.ACTIONS[np.argmax(action_returns)]
                    policy[i, j] = new_action
                    if policy_stable and old_action != new_action:
                        policy_stable = False
            print(f"Policy stable {policy_stable}")
            if policy_stable:
                fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
                fig.set_ylabel("# cars at first location", fontsize=30)
                fig.set_yticks(list(reversed(range(self.MAX_CARS + 1))))
                fig.set_xlabel("# cars at second location", fontsize=30)
                fig.set_title("optimal value", fontsize=30)
                break

            iterations += 1

        plt.savefig("Yeeet_figure_4_2.png")
        plt.close()


if __name__ == "__main__":
    jacks = JacksCarRental(max_cars=20, max_move_of_cars=5)
    ENV = GENV("4-Jacks")
    ENV.createResDir()
    kv, kp = jacks.policy_iteration()
    ENV.save_value_iter(kv, "Value functions", with_steps=True)
    ENV.save_value_iter(kp, "Policies", with_steps=True)
