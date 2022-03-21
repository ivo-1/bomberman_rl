import os
from collections import deque, namedtuple
from statistics import (
    avg_seen_actions,
    distribution_of_best_actions,
    fraction_of_unseen_states,
)
from typing import List

import numpy as np

import events as e
from agent_code.coli_agent.callbacks import ACTIONS, state_to_features
from agent_code.coli_agent.plots import get_plots

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions

# --- Custom Events ---

FOLLOWED_COIN_DIRECTION = (
    "FOLLOWED_COIN_DIRECTION"  # went in direction indicated by coin/crate feature
)
NOT_FOLLOWED_COIN_DIRECTION = "NOT_FOLLOWED_COIN_DIRECTION"

FOLLOWED_BOMB_DIRECTION = "FOLLOWED_BOMB_DIRECTION"  # went in direction indicated by bomb feature
NOT_FOLLOWED_BOMB_DIRECTION = "NOT_FOLLOWED_BOMB_DIRECTION"

DROPPED_BAD_BOMB = (
    "DROPPED_BAD_BOMB"  # tried to drop bomb when impossible or dropped bomb when dangerous
)
DROPPED_UNNECESSARY_BOMB = (
    "DROPPED_UNNECESSARY_BOMB"  # drop bomb when there weren't any crates or enemies
)

TARGETED_MANY_CRATES = "TARGETED_MANY_CRATES"
TARGETED_SOME_CRATES = "TARGETED_SOME_CRATES"
TARGETED_ENEMY = "TARGETED_ENEMY"

BLOCKED = "BLOCKED"


def setup_training(self):
    """Sets up training"""
    self.exploration_rate = self.exploration_rate_initial
    self.learning_rate = 0.5
    self.discount_rate = 0.2

    # (s, a, s', r)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode = 0  # need to keep track of episodes
    self.rewards_of_episode = 0


def game_events_occurred(self, old_game_state, self_action: str, new_game_state, events):
    """Called once after each time step (after act()) except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.

    Will call state_to_features, and can then use these features for adding our custom events.
    (if features = ... -> events.append(OUR_EVENT)). But events can also be added independently of features,
    just using game state in general. Leveraging of features more just to avoid code duplication.
    """
    self.history.append(new_game_state["self"][-1])

    old_state = self.old_state
    self.new_state = state_to_features(self, new_game_state)
    new_state = self.new_state

    old_feature_dict = self.state_list[old_state]

    # Custom events and stuff

    reward_coin_feature = True
    if (
        old_feature_dict["bomb_safety_direction"] != "CLEAR"
        or old_feature_dict["in_enemy_zone"] != 0
    ):
        reward_coin_feature = False
    # somestimes coin_direction is random and can, e.g., be an explosion
    if old_feature_dict["coin_direction"] == "UP" and old_feature_dict["up"] != "BLOCKED":
        reward_coin_feature = False
    if old_feature_dict["coin_direction"] == "DOWN" and old_feature_dict["down"] != "BLOCKED":
        reward_coin_feature = False
    if old_feature_dict["coin_direction"] == "RIGHT" and old_feature_dict["right"] != "BLOCKED":
        reward_coin_feature = False
    if old_feature_dict["coin_direction"] == "LEFT" and old_feature_dict["left"] != "BLOCKED":
        reward_coin_feature = False

    if reward_coin_feature is True:
        if old_feature_dict["coin_direction"] == self_action:
            events.append(FOLLOWED_COIN_DIRECTION)
        else:
            events.append(NOT_FOLLOWED_COIN_DIRECTION)

    if old_feature_dict["bomb_safety_direction"] != "CLEAR":
        if self_action == old_feature_dict["bomb_safety_direction"]:
            events.append(FOLLOWED_BOMB_DIRECTION)
        else:
            events.append(NOT_FOLLOWED_BOMB_DIRECTION)

    if old_feature_dict["safe_to_bomb"] == 0 and self_action == "BOMB":
        events.append(DROPPED_BAD_BOMB)

    if old_feature_dict["safe_to_bomb"] == 1:
        if self_action == "BOMB":
            if old_feature_dict["crate_priority"] == "HIGH":
                events.append(TARGETED_MANY_CRATES)
            elif old_feature_dict["crate_priority"] == "LOW":
                events.append(TARGETED_SOME_CRATES)
            elif old_feature_dict["in_enemy_zone"] == 1:
                events.append(TARGETED_ENEMY)
            else:
                events.append(DROPPED_UNNECESSARY_BOMB)

    if old_feature_dict["up"] == "BLOCKED" and self_action == "UP":
        events.append(BLOCKED)
    elif old_feature_dict["down"] == "BLOCKED" and self_action == "DOWN":
        events.append(BLOCKED)
    elif old_feature_dict["right"] == "BLOCKED" and self_action == "RIGHT":
        events.append(BLOCKED)
    elif old_feature_dict["left"] == "BLOCKED" and self_action == "LEFT":
        events.append(BLOCKED)

    self.logger.debug(f'Old coords: {old_game_state["self"][3]}')
    self.logger.debug(f'New coords: {new_game_state["self"][3]}')
    self.logger.debug(f"Action: {self_action}")

    # collect reward
    reward = reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(
            old_state,
            self_action,
            new_state,
            reward,
        )
    )

    action_idx = ACTIONS.index(self_action)
    self.logger.debug(f"Action index chosen: {action_idx}")

    self.rewards_of_episode += reward
    self.logger.debug(f"Old Q-Table old state: {self.q_table[old_state]}")

    self.q_table[old_state, action_idx] = self.q_table[
        old_state, action_idx
    ] + self.learning_rate * (
        reward
        + self.discount_rate * np.max(self.q_table[new_state])
        - self.q_table[old_state, action_idx]
    )
    self.logger.debug(f"New Q-Table old state: {self.q_table[old_state]}")

    self.logger.info(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}\n\
        ==============================================================================================\
        =============================================================================================='
    )


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent after the last step of a round."""
    self.transitions.append(
        Transition(
            state_to_features(self, last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )
    self.rewards_of_episode += self.transitions[-1][3]

    self.logger.info(f"Total rewards in episode {self.episode}: {self.rewards_of_episode}")
    self.logger.info(f"Final Score: {last_game_state['self'][1]}")

    self.exploration_rate = self.exploration_rate_end + (
        self.exploration_rate_initial - self.exploration_rate_end
    ) * np.exp(
        -self.exploration_decay_rate * self.episode
    )  # decay

    q_table_fraction_unseen_current = fraction_of_unseen_states(self.q_table)
    q_table_average_seen_current = avg_seen_actions(self.q_table)
    q_table_distribution_of_actions_current = distribution_of_best_actions(self.q_table)

    self.logger.info(f"Fraction of unseen states: {q_table_fraction_unseen_current}")
    self.logger.info(f"Average seen actions per state: {q_table_average_seen_current}")
    self.logger.info(
        f"Distribution of actions over all states: {q_table_distribution_of_actions_current}"
    )
    self.q_table_fraction_unseen.append(q_table_fraction_unseen_current)
    self.q_table_average_seen.append(q_table_average_seen_current)

    self.exploration_rates_of_episodes.append(self.exploration_rate)
    self.rewards_of_episodes.append(self.rewards_of_episode)
    self.game_scores_of_episodes.append(last_game_state["self"][1])

    if self.episode % 250 == 0 and self.episode != 0:
        self.logger.info(f"Saving Q-Table at episode: {self.episode}")
        np.save(os.path.join("q_tables", f"q_table-{self.timestamp}"), self.q_table)

        self.logger.info(f"Creating plots *after* episode {self.episode}...")
        self.q_table_distribution_of_actions.append(q_table_distribution_of_actions_current)
        get_plots(self)

    self.rewards_of_episode = 0
    self.episode += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    Returns a summed up reward/penalty for a given list of events that happened.
    """

    game_rewards = {
        # e.BOMB_DROPPED: 10,
        e.BOMB_EXPLODED: 0,
        # e.COIN_COLLECTED: 100,
        e.COIN_FOUND: 0,
        e.WAITED: 0,
        e.CRATE_DESTROYED: 0,
        e.GOT_KILLED: 0,
        e.KILLED_OPPONENT: 0,
        # e.KILLED_SELF: -1000,  # this *also* triggers GOT_KILLED
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,
        e.INVALID_ACTION: 0,
        # FOLLOWED_COIN_DIRECTION: 15,
        # NOT_FOLLOWED_COIN_DIRECTION: -20,
        FOLLOWED_BOMB_DIRECTION: 25,
        NOT_FOLLOWED_BOMB_DIRECTION: -50,
        DROPPED_BAD_BOMB: -100,
        DROPPED_UNNECESSARY_BOMB: -75,
        TARGETED_MANY_CRATES: 50,
        TARGETED_SOME_CRATES: 10,
        # TARGETED_ENEMY: 30,
        BLOCKED: -100,
        # e.MOVED_DOWN: 25,
        # e.MOVED_LEFT: 25,
        # e.MOVED_RIGHT: 25,
        # e.MOVED_UP: 25
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
