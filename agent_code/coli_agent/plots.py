import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_q_table_stats(self, x, fraction_unseen, average_seen, distribution_of_actions) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=150)  # create figure & 3 axes
    fig.suptitle(f"Q-Table Stats after episode {self.episode}")

    fraction_unseen_hat = savgol_filter(fraction_unseen, 10, 2)
    axs[0].set_title("Fraction of unseen states in Q-Table")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Fraction of unseen states")
    axs[0].set_xlim([0, max(x)])
    axs[0].plot(x, fraction_unseen, label="raw")
    axs[0].plot(x, fraction_unseen_hat, color="red", label="smoothed")
    axs[0].legend()

    average_seen_hat = savgol_filter(average_seen, 10, 2)
    axs[1].set_title("Average seen actions in Q-Table per state")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average seen actions")
    axs[1].set_xlim([0, max(x)])
    axs[1].plot(x, average_seen, label="raw")
    axs[1].plot(x, average_seen_hat, color="red", label="smoothed")
    axs[1].legend()

    axs[2].set_title("Average distribution of actions over all states")
    axs[2].set_xlabel("Action")
    axs[2].set_ylabel("Probability")
    axs[2].bar(distribution_of_actions.keys(), distribution_of_actions.values())

    fig.tight_layout()
    fig.savefig(f"plots/q-table-plots_{self.timestamp}.png")  # , bbox_inches='tight')
    plt.close(fig)


def plot_exploration_rate(self, x, exploration_rates) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=150)  # create figure & 3 axes

    ax.set_title("Exploration rate over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration rate")
    ax.set_xlim([0, max(x)])
    ax.plot(x, exploration_rates)

    fig.tight_layout()
    fig.savefig(f"plots/exploration-rate_{self.timestamp}.png")  # , bbox_inches='tight')
    plt.close(fig)


def plot_rewards(self, x, rewards_of_episodes) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=150)  # create figure & 3 axes

    rewards_of_episodes_hat = savgol_filter(rewards_of_episodes, 10, 2)
    ax.set_title("Rewards over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_xlim([0, max(x)])
    ax.plot(x, rewards_of_episodes, label="raw")
    ax.plot(x, rewards_of_episodes_hat, color="red", label="smoothed")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"plots/rewards_{self.timestamp}.png")  # , bbox_inches='tight')
    plt.close(fig)


def plot_game_score(self, x, game_scores_of_episodes) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=150)  # create figure & 3 axes

    game_scores_of_episodes_hat = savgol_filter(game_scores_of_episodes, 10, 2)
    ax.set_title("Game scores over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Game Score")
    ax.set_xlim([0, max(x)])
    ax.plot(x, game_scores_of_episodes, label="raw")
    ax.plot(x, game_scores_of_episodes_hat, color="red", label="smoothed")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"plots/game-scores_{self.timestamp}.png")  # , bbox_inches='tight')
    plt.close(fig)


def get_plots(self) -> None:
    x = range(self.episode + 1)
    plot_q_table_stats(
        self,
        x,
        self.q_table_fraction_unseen,
        self.q_table_average_seen,
        self.q_table_distribution_of_actions[-1],
    )
    plot_exploration_rate(self, x, self.exploration_rates_of_episodes)
    plot_rewards(self, x, self.rewards_of_episodes)
    plot_game_score(self, x, self.game_scores_of_episodes)
