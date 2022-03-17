import argparse

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

parser = argparse.ArgumentParser(description="Calculates stats from a given q-table")
parser.add_argument("q_table", type=str, help="Path to the q-table you want to calculate stats for")


def fraction_of_unseen_states(q_table: np.array) -> float:
    """Returns the fraction of unseen states in the q-table between 0 and 1.
    A state is considered unseen if the row of that state is all 0."""
    action_count_per_state = np.count_nonzero(
        q_table, axis=1
    )  # counts how many actions have been seen (non-zero) per state
    return 1 - np.count_nonzero(action_count_per_state) / len(action_count_per_state)


def avg_seen_actions(q_table: np.array) -> float:
    """Returns the average seen number of actions per state"""
    action_count_per_state = np.count_nonzero(q_table, axis=1)
    return np.average(action_count_per_state)


def distribution_of_best_actions(q_table: np.array) -> dict:
    best_actions_per_state = list(np.argmax(q_table, axis=1))
    distribution = {}
    for action in ACTIONS:
        action_idx = ACTIONS.index(action)
        distribution[action] = best_actions_per_state.count(action_idx) / len(
            best_actions_per_state
        )

    return distribution


if __name__ == "__main__":
    args = parser.parse_args()
    q_table = np.load(args.q_table)

    print(f"Fraction of unseen states: {fraction_of_unseen_states(q_table)}")
    print(f"Average seen actions per state: {avg_seen_actions(q_table)}")
    print(f"Distribution of actions over all states: {distribution_of_best_actions(q_table)}")
