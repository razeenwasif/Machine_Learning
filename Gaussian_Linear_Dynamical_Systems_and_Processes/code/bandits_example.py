import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from bandits import BetaBernoulliBandit


def run_bandit_simulation(
    bandit, num_steps: int, strategy: str = "thompson"
) -> Dict:
    """
    Run a bandit simulation.

    Parameters:
    -----------
    bandit: BetaBernoulliBandit
        The bandit to run the simulation on
    num_steps: int
        Number of steps to run the simulation for
    strategy: str
        The strategy to use ('random', 'epsilon_greedy', or 'thompson')

    Returns:
    --------
    results: Dict
        Dictionary containing the results of the simulation
    """
    rewards = []
    chosen_arms = []

    for t in range(num_steps):
        # Choose an arm based on the strategy
        if strategy == "random":
            arm = np.random.randint(0, bandit.num_arms)
        elif strategy == "epsilon_greedy":
            arm = bandit.choose_arm_epsilon_greedy(epsilon=0.1)
        elif strategy == "thompson":
            arm = bandit.choose_arm_thompson_sampling()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Pull the arm and get reward
        reward = bandit.pull_arm(arm)

        # Update posterior
        bandit.update_posterior(arm, reward)

        # Record results
        rewards.append(reward)
        chosen_arms.append(arm)

    return {
        "rewards": np.array(rewards),
        "chosen_arms": np.array(chosen_arms),
        "cumulative_reward": np.cumsum(rewards),
        "bandit": bandit,
    }


def plot_comparison(
    results_list: List[Dict], strategy_names: List[str], title: str = None
) -> None:
    """
    Plot a comparison of multiple bandit simulation results.

    Parameters:
    -----------
    results_list: List[Dict]
        List of dictionaries containing the results of multiple simulations
    strategy_names: List[str]
        Names of the strategies used
    title: str
        Title for the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    colors = plt.cm.tab10.colors

    # Plot cumulative rewards
    for i, (results, name) in enumerate(zip(results_list, strategy_names)):
        axes[0].plot(results["cumulative_reward"], label=name, color=colors[i])
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Rewards so far (e.g., no of patients treated)")
    axes[0].set_title("Cumulative Reward")
    axes[0].legend()

    # Plot arm selection frequency
    width = 0.2
    x = np.arange(results_list[0]["bandit"].num_arms)

    for i, (results, name) in enumerate(zip(results_list, strategy_names)):
        arm_counts = np.bincount(
            results["chosen_arms"], minlength=results_list[0]["bandit"].num_arms
        )
        arm_freq = arm_counts / len(results["chosen_arms"])
        axes[1].bar(
            x + i * width - (len(results_list) - 1) * width / 2,
            arm_freq,
            width,
            label=name,
            color=colors[i],
        )

    axes[1].set_xlabel("Arm")
    axes[1].set_ylabel("Selection Frequency")
    axes[1].set_title("Arm Selection Frequency")
    axes[1].set_xticks(x)
    axes[1].legend()

    if title:
        fig.suptitle(title, fontsize=16)

    # plt.savefig(
    #     "/tmp/bandits_comparison.pdf",
    #     bbox_inches="tight",
    #     dpi=300,
    #     pad_inches=0,
    # )
    plt.show()


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Create a Bernoulli bandit with 2 arms
    num_arms = 2
    true_probs = [0.5, 0.6]  # Arm 2 is better
    num_steps = 2000

    # Run simulations with different strategies
    random_results = run_bandit_simulation(
        BetaBernoulliBandit(num_arms=num_arms, true_probs=true_probs),
        num_steps=num_steps,
        strategy="random",
    )

    # Thompson sampling simulation
    thompson_results = run_bandit_simulation(
        BetaBernoulliBandit(num_arms=num_arms, true_probs=true_probs),
        num_steps=num_steps,
        strategy="thompson",
    )

    # Epsilon-greedy simulation
    epsilon_greedy_results = run_bandit_simulation(
        BetaBernoulliBandit(num_arms=num_arms, true_probs=true_probs),
        num_steps=num_steps,
        strategy="epsilon_greedy",
    )

    # Plot comparison
    plot_comparison(
        [random_results, thompson_results, epsilon_greedy_results],
        ["Random", "Thompson Sampling", "Epsilon-Greedy"],
    )
