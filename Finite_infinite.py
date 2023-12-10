import numpy as np
from scipy.integrate import odeint


class InfiniteInventoryMDP:
    def __init__(
        self,
        holding_cost_rate,
        shortage_cost_rate,
        ordering_cost_rate,
        initial_inventory,
        unit_order_quantity,
        discount_factor,
    ):
        self.holding_cost_rate = holding_cost_rate
        self.shortage_cost_rate = shortage_cost_rate
        self.ordering_cost_rate = ordering_cost_rate
        self.initial_inventory = initial_inventory
        self.unit_order_quantity = unit_order_quantity
        self.discount_factor = discount_factor

    def continuous_time_dynamics(self, state, t):
        def demand_rate(t):
            return 15 + 5 * np.sin(0.2 * t)

        inventory_level, order_in_transit = state
        demand = demand_rate(t)

        d_inventory_dt = demand - min(inventory_level, demand) + order_in_transit
        d_order_in_transit_dt = max(0, demand - inventory_level)

        return [d_inventory_dt, d_order_in_transit_dt]

    def rewards(self, state):
        inventory_level, _ = state
        holding_cost = self.holding_cost_rate * max(0, inventory_level)
        shortage_cost = self.shortage_cost_rate * max(0, -inventory_level)
        return -holding_cost - shortage_cost

    def value_iteration(self, time_horizon, num_time_steps):
        initial_state = [self.initial_inventory, 0]

        def ode_system(state, t):
            return self.continuous_time_dynamics(state, t)

        time_points = np.linspace(0, time_horizon, num_time_steps)
        result = odeint(ode_system, initial_state, time_points)

        V = np.zeros(len(time_points))
        for i, state in enumerate(result):
            V[i] = sum(
                self.rewards(state) * np.exp(-self.discount_factor * (time_horizon - t))
                for t in time_points[i:]
            )

        return time_points, V


# Example Usage:
holding_cost_rate = 0.1
shortage_cost_rate = 1.0
ordering_cost_rate = 2.0
initial_inventory = 20
unit_order_quantity = 30


discount_factor = 0.05

inventory_mdp = InfiniteInventoryMDP(
    holding_cost_rate,
    shortage_cost_rate,
    ordering_cost_rate,
    initial_inventory,
    unit_order_quantity,
    discount_factor,
)
time_points, value_function = inventory_mdp.value_iteration(
    time_horizon=50, num_time_steps=100
)
print("Optimal Value Function over Time:")
print(value_function)
