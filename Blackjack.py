import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class BlackJack:
    def __init__(self) -> None:
        self.HIT = 0
        self.STAND = 1
        self.ACTIONS = [self.HIT, self.STAND]

    def get_card(self):
        card = np.random.randint(1, 14)
        card = min(card, 10)
        return card

    def card_value(self, card):
        return 11 if card == 1 else card

    def init_policy(self, dealer=False):
        if not dealer:
            policy = np.zeros(22)
            for i in range(12, 20):
                policy[i] = self.HIT
            policy[20] = self.STAND
            policy[21] = self.STAND
        else:
            policy = np.zeros(22)
            for i in range(12, 17):
                policy[i] = self.HIT
            for i in range(17, 22):
                policy[i] = self.STAND
        return policy

    def reset_play(self, initial_state=None):
        player_sum = 0

        dealer_card1 = 0
        dealer_card2 = 0

        ace_player = False  # Set usable aces to false
        ace_dealer = False

        if initial_state is not None:
            ace_player, player_sum, dealer_card1 = initial_state
        # Always get to 12 atlease

        while player_sum < 12:
            # Get card
            card = self.get_card()
            player_sum += self.card_value(card)
            if player_sum > 21:
                assert player_sum == 22
                # Last card must be ace
                player_sum -= 10
            else:
                ace_player |= 1 == card

        dealer_card1 = self.get_card()
        dealer_card2 = self.get_card()
        dealer_sum = self.card_value(dealer_card1) + self.card_value(dealer_card2)
        ace_dealer = 1 in (dealer_card1, dealer_card2)
        if dealer_sum > 21:
            assert dealer_sum == 22
            # Use ace
            dealer_sum -= 10
        assert dealer_sum <= 21
        assert player_sum <= 21 and player_sum >= 12

        state = [ace_player, player_sum, dealer_card1]
        return state, dealer_sum, ace_dealer

    def reward(self, dealer_sum, player_sum):
        assert player_sum <= 21 and dealer_sum <= 21
        if player_sum > dealer_sum:
            return 1
        elif player_sum == dealer_sum:
            return 0
        else:
            return -1

    def play(self, player_policy=None, initial_state=None, initial_action=None):
        # init policies
        if player_policy is None:
            player_policy = self.init_policy(dealer=False)
        else:
            player_policy = player_policy
        dealer_policy = self.init_policy(dealer=True)

        player_history = []
        state, dealer_sum, ace_dealer = self.reset_play()
        ace_player, player_sum, dealer_card1 = state
        # player turn
        while True:
            if initial_action is not None:
                action = initial_action
                initial_action = None
            else:
                action = player_policy(ace_player, player_sum, dealer_card1)

            player_history.append((ace_player, player_sum, dealer_card1, action))

            if action == self.STAND:
                break
            # Get new card
            card = self.get_card()

            ace_count = int(ace_player)
            if card == 1:
                ace_count += 1
            player_sum += self.card_value(card)
            while player_sum > 21 and ace_count:
                player_sum -= 10
                ace_count -= 1
            if player_sum > 21:
                return state, -1, player_history
            assert player_sum <= 21
            ace_player = ace_count == 1

            # Dealer turn
        # Dealer turn
        while True:
            action = dealer_policy[dealer_sum]
            if action == self.STAND:
                break
            new_card = self.get_card()
            ace_count = int(ace_dealer)
            if new_card == 1:
                ace_count += 1
            dealer_sum += self.card_value(new_card)
            while dealer_sum > 21 and ace_count:
                dealer_sum -= 10
                ace_count -= 1
            if dealer_sum > 21:
                return state, 1, player_history
            ace_dealer = ace_count == 1

        # compare
        assert player_sum <= 21 and dealer_sum
        reward = self.reward(dealer_sum, player_sum)

        # Move the return statement outside the loop
        return state, reward, player_history

    def monte_carlo_on_policy(self, episodes):
        """
        Estimating the state-value function by simulation
        """
        states_usable_ace = np.zeros((10, 10))

        # Initialize counts to 1 to avoid 0 div.
        states_usable_ace_count = np.ones((10, 10))

        states_no_usable_ace = np.zeros((10, 10))

        states_no_usable_ace_count = np.ones((10, 10))
        for i in tqdm(range(0, episodes), desc=f"Simulating episode: {episodes}"):
            state, reward, player_history = self.play()
            for ace_player, player_sum, dealer_card, _ in player_history:
                player_sum -= 12
                dealer_card -= 1
                if ace_player:
                    states_usable_ace_count[player_sum, dealer_card] += 1
                    states_usable_ace[player_sum, dealer_card] += reward
                else:
                    states_no_usable_ace_count[player_sum, dealer_card] += 1
                    states_no_usable_ace[player_sum, dealer_card] += reward
        return (
            states_no_usable_ace / states_usable_ace_count,
            states_no_usable_ace / states_no_usable_ace_count,
        )

    def montecarlo_first_visit(self, episodes, policy_to_evaluate=None):
        # Init arbitrary values
        states_values_usable_ace = np.zeros((10, 10))
        states_counts_usable_ace = np.ones((10, 10))

        states_values_no_usable_ace = np.zeros((10, 10))
        states_counts_no_usable_ace = np.ones((10, 10))

        for episode in tqdm(range(episodes)):
            state, reward, player_history = self.play()

            visited_states = set()

            G = 0
            for t in reversed(range(len(player_history))):
                ace_player, player_sum, dealer_card, action = player_history[t]
                player_sum -= 12
                dealer_card -= 1

                G += reward

                if (ace_player, player_sum, dealer_card) not in visited_states:
                    visited_states.add((ace_player, player_sum, dealer_card))

                    # Update state values based on usable ace
                    if ace_player:
                        states_counts_usable_ace[player_sum, dealer_card] += 1
                        states_values_usable_ace[player_sum, dealer_card] += G
                    else:
                        states_counts_no_usable_ace[player_sum, dealer_card] += 1
                        states_values_no_usable_ace[player_sum, dealer_card] += G

        # Calculate average state values
        states_values_usable_ace /= states_counts_usable_ace
        states_values_no_usable_ace /= states_counts_no_usable_ace

        return states_values_usable_ace, states_values_no_usable_ace

    # function form of target policy of player
    def target_policy_player(self, player_sum, POLICY_PLAYER):
        return POLICY_PLAYER[player_sum]

    # function form of behavior policy of player
    def behavior_policy_player(self, usable_ace_player, player_sum, dealer_card):
        if np.random.binomial(1, 0.5) == 1:
            return self.STAND
        return self.HIT

    def MSES(self, episodes):
        Q = np.zeros(
            (10, 10, 2, 2)
        )  # state action value state = (sum, card, usable ace) action
        QCounter = np.ones((10, 10, 2, 2))

        # Random policy
        def behavior_policy(usable_ace, player_sum, dealer_card):
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            # argmax
            values = (
                Q[player_sum, dealer_card, usable_ace, :]
                / QCounter[player_sum, dealer_card, usable_ace, :]
            )
            return np.random.choice(
                [
                    action_
                    for action_, value_ in enumerate(values)
                    if value_ == np.max(values)
                ]
            )

        def s0A0random():
            initial_state = [
                bool(np.random.choice([0, 1])),
                np.random.choice(range(12, 22)),
                np.random.choice(range(1, 11)),
            ]
            initial_action = np.random.choice(self.ACTIONS)
            return initial_state, initial_action

        # Q learning
        for episode in tqdm(range(episodes)):
            initial_state, initial_action = s0A0random()
            # Force exploring at the beginning
            if episode:
                current_policy = behavior_policy
            else:
                current_policy = self.target_policy_player
            state, reward, player_history = self.play(
                current_policy, initial_state, initial_action
            )

            first_visit_check = set()
            for ace, player_sum, dealer_card, action in player_history:
                ace = int(ace)
                player_sum -= 12
                dealer_card -= 1
                state_action = (player_sum, dealer_card, ace, action)
                if state_action in first_visit_check:
                    continue
                first_visit_check.add(state_action)
                QCounter[player_sum, dealer_card, ace, action] += 1
                Q[player_sum, dealer_card, ace, action] += reward

            # Get averages
        return Q / QCounter

    def monte_carlo_exploring_starts_comparison(self):
        Q = self.MSES(episodes=500000)

        Q_no_usable_ace = np.max(Q[:, :, 0, :], axis=-1)
        Q_usable_ace = np.max(Q[:, :, 1, :], axis=-1)

        action_no_usable_ace = np.argmax(Q[:, :, 0, :], axis=-1)
        action_usable_ace = np.argmax(Q[:, :, 1, :], axis=-1)

        images = [
            action_usable_ace,
            Q_usable_ace,
            action_no_usable_ace,
            Q_no_usable_ace,
        ]
        titles = [
            "Optimal policy with usable Ace",
            "Optimal value with usable Ace",
            "Optimal policy without usable Ace",
            "Optimal value without usable Ace",
        ]

        _, axes = plt.subplots(2, 2, figsize=(40, 30))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        for image, title, axis in zip(images, titles, axes):
            fig = sns.heatmap(
                np.flipud(image),
                cmap="YlGnBu",
                ax=axis,
                xticklabels=range(1, 11),
                yticklabels=list(reversed(range(12, 22))),
            )
            fig.set_ylabel("player sum", fontsize=30)
            fig.set_xlabel("dealer showing", fontsize=30)
            fig.set_title(title, fontsize=30)

        plt.show()

    def monte_carlo_on_policy_comparison(self, algorithm):
        if algorithm == "ON_POLICY":
            states_usable_ace_0, states_no_usable_ace0 = self.monte_carlo_on_policy(
                episodes=5000
            )
            states_usable_ace_1, states_no_usable_ace1 = self.monte_carlo_on_policy(
                episodes=100000
            )
            states_usable_ace_2, state_no_usable_ace2 = self.monte_carlo_on_policy(
                episodes=500000
            )
        if algorithm == "FIRST_VISITS":
            states_usable_ace_0, states_no_usable_ace0 = self.montecarlo_first_visit(
                episodes=5000
            )
            states_usable_ace_1, states_no_usable_ace1 = self.montecarlo_first_visit(
                episodes=100000
            )
            states_usable_ace_2, state_no_usable_ace2 = self.montecarlo_first_visit(
                episodes=500000
            )

        states = [
            states_usable_ace_0,
            states_usable_ace_1,
            states_usable_ace_2,
            states_no_usable_ace0,
            states_no_usable_ace1,
            state_no_usable_ace2,
        ]
        titles = [
            "Usable Ace, 5000 Episodes",
            "Usable Ace, 10000 Episodes",
            "Usable Ace, 500000 Episodes",
            "No Usable Ace, 5000 Episodes",
            "No Usable Ace, 10000 Episodes",
            "No Usable Ace, 500000 Episodes",
        ]

        _, axes = plt.subplots(2, 3, figsize=(40, 30))
        axes = axes.flatten()

        for state, title, axis in zip(states, titles, axes):
            fig = sns.heatmap(
                np.flipud(state),
                cmap="YlGnBu",
                ax=axis,
                xticklabels=range(1, 11),
                yticklabels=list(reversed(range(12, 22))),
            )
            fig.set_ylabel("player sum", fontsize=30)
            fig.set_xlabel("dealer showing", fontsize=30)
            fig.set_title(title, fontsize=30)
        plt.show()


if __name__ == "__main__":
    bj = BlackJack()
    # bj.monte_carlo_on_policy_comparison(algorithm="FIRST_VISITS")
    bj.MSES(episodes=10000)
    bj.monte_carlo_exploring_starts_comparison()
