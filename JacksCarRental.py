import numpy as np
from scipy.stats import poisson
from tqdm import trange
from OSHelper import GENV


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
        v = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        policy = np.zeros_like(v)
        delta = 1e-4
        kPolicies = []
        change = 0
        # Policy evaluation
        while True:
            V = v.copy()
            # Loop through all states!
            for i in trange(
                self.MAX_CARS + 1, desc=f"Policy Evaluation, last change was {change}"
            ):
                for j in range(self.MAX_CARS + 1):
                    returns = self.step([i, j], policy[i, j], v)
                    v[i, j] = returns
            change = np.abs(v - V).max()
            if np.sum(np.abs(v - V)) < delta:
                break
        policy_stable = True
        # Policy improvement
        kPolicies.append(policy)
        for i in trange(self.MAX_CARS + 1, desc="Policy Improvement"):
            for j in range(self.MAX_CARS + 1):
                old_action = policy[i, j]
                action_value_returns = []
                for action in self.ACTIONS:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        returns = self.step([i, j], policy[i, j], v)
                        action_value_returns.append(returns)
                    else:
                        action_value_returns.append(-np.inf)
                    best_action = self.ACTIONS[np.argmax(action_value_returns)]
                    policy[i, j] = best_action
                    if policy_stable and old_action != best_action:
                        policy_stable = False
        return policy, v, kPolicies


if __name__ == "__main__":
    jacks = JacksCarRental()
    policy, v, kPolicies = jacks.policy_iteration()

    env = GENV("4-Jack")
    env.createResDir()
    env.save_value_iter(kPolicies, "Jacks car rental - Policies", with_steps=True)
    env.save_value_iter(v, "Jacks car rental - Value function", with_steps=False)
