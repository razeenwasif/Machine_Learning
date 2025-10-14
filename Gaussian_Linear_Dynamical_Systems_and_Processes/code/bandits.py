import numpy as np
from typing import Tuple, List, Optional


class BetaBernoulliBandit:
    """
    Implementation of a Beta-Bernoulli Bandit model.
    """

    def __init__(self, num_arms: int, true_probs: Optional[List[float]] = None):
        """
        Initialize the bandit.

        Parameters:
        -----------
        num_arms: int
            Number of arms (actions)
        true_probs: Optional[List[float]]
            True success probabilities for each arm (for simulation)
        """
        self.num_arms = num_arms

        # True probabilities (unknown in practice, used for simulation)
        if true_probs is None:
            self.true_probs = np.random.uniform(0, 1, size=num_arms)
        else:
            self.true_probs = np.array(true_probs)

        # Initialize Beta prior parameters for each arm
        # Beta(1, 1) is a uniform prior over [0, 1]
        self.alpha = np.ones(num_arms)  # number of successes + 1
        self.beta = np.ones(num_arms)  # number of failures + 1

        # Store history of arm pulls and rewards
        self.arm_history = []
        self.reward_history = []

    def update_posterior(self, arm: int, reward: int) -> None:
        """
        Update the posterior distribution for the selected arm.

        Parameters:
        -----------
        arm: int
            The arm that was pulled
        reward: int
            The reward received (0 or 1)
        """
        # Store the arm pull and reward in history
        self.arm_history.append(arm)
        self.reward_history.append(reward)

        # TODO: Implement the posterior update (Question 2.2)
        # Your implementation here

    def get_arm_posterior(self, arm: int) -> Tuple[float, float]:
        """
        Get the current posterior distribution parameters for an arm.

        Parameters:
        -----------
        arm: int
            The arm to get the posterior for

        Returns:
        --------
        alpha: float
            Alpha parameter of the Beta distribution
        beta: float
            Beta parameter of the Beta distribution
        """
        return self.alpha[arm], self.beta[arm]

    def get_arm_mean(self, arm: int) -> float:
        """
        Get the mean of the current posterior distribution for an arm.

        Parameters:
        -----------
        arm: int
            The arm to get the mean for

        Returns:
        --------
        mean: float
            Mean of the posterior distribution
        """
        # For Beta distribution, mean = alpha / (alpha + beta)
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def pull_arm(self, arm: int) -> int:
        """
        Simulate pulling an arm and receiving a reward.

        Parameters:
        -----------
        arm: int
            The arm to pull

        Returns:
        --------
        reward: int
            The reward received (0 or 1)
        """
        # Generate a Bernoulli reward based on the true probability
        return np.random.binomial(1, self.true_probs[arm])

    def sample_from_posterior(self, arm: int) -> float:
        """
        Sample a value from the posterior distribution of an arm.

        Parameters:
        -----------
        arm: int
            The arm to sample from

        Returns:
        --------
        sample: float
            A sample from the posterior distribution
        """
        return np.random.beta(self.alpha[arm], self.beta[arm])

    def choose_arm_thompson_sampling(self) -> int:
        """
        Choose an arm using Thompson sampling.

        Returns:
        --------
        arm: int
            The arm to pull
        """
        # Sample from each arm's posterior
        samples = np.array(
            [self.sample_from_posterior(arm) for arm in range(self.num_arms)]
        )

        # Choose the arm with the highest sample
        return np.argmax(samples)

    def choose_arm_epsilon_greedy(self, epsilon: float = 0.1) -> int:
        """
        Choose an arm using epsilon-greedy strategy.

        Parameters:
        -----------
        epsilon: float
            Probability of choosing a random arm

        Returns:
        --------
        arm: int
            The arm to pull
        """
        if np.random.random() < epsilon:
            # Explore: choose a random arm
            return np.random.randint(0, self.num_arms)
        else:
            # Exploit: choose the arm with the highest expected reward
            expected_rewards = self.alpha / (self.alpha + self.beta)
            return np.argmax(expected_rewards)
