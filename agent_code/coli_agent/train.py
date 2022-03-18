from collections import deque, namedtuple
from statistics import (
    avg_seen_actions,
    distribution_of_best_actions,
    fraction_of_unseen_states,
)
from typing import List

import numpy as np

import events as e
from agent_code.coli_agent.callbacks import (
    ACTIONS,
    get_neighboring_tiles_until_wall,
    state_to_features,
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions

# --- Custom Events ---

WAS_BLOCKED = "WAS_BLOCKED"  # tried to move into a wall/crate/enemy/explosion (strong penalty)
MOVED = "MOVED"  # moved somewhere (and wasn't blocked)

FOLLOWED_DIRECTION = "FOLLOWED_DIRECTION"  # went in direction indicated by coin/crate feature
NOT_FOLLOWED_DIRECTION = "NOT_FOLLOWED_DIRECTION"

INCREASED_SURROUNDING_CRATES = (
    "INCREASED_SURROUNDING_CRATES"  # increased or stayed the same; low reward
)
DECREASED_SURROUNDING_CRATES = (
    "DECREASED_SURROUNDING_CRATES"  # equal or slightly higher penalty for balance
)


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
    new_feature_dict = self.state_list[new_state]

    # Custom events and stuff

    if old_feature_dict["blocked_down"] == 1 and self_action == "DOWN":
        events.append(WAS_BLOCKED)
    elif old_feature_dict["blocked_up"] == 1 and self_action == "UP":
        events.append(WAS_BLOCKED)
    elif old_feature_dict["blocked_right"] == 1 and self_action == "RIGHT":
        events.append(WAS_BLOCKED)
    elif old_feature_dict["blocked_down"] == 1 and self_action == "LEFT":
        events.append(WAS_BLOCKED)

    old_neighbors = get_neighboring_tiles_until_wall(
        old_game_state["self"][-1], 3, game_state=old_game_state
    )
    new_neighbors = get_neighboring_tiles_until_wall(
        new_game_state["self"][-1], 3, game_state=new_game_state
    )

    if old_feature_dict["coin_direction"] == self_action:
        events.append(FOLLOWED_DIRECTION)
    else:
        events.append(NOT_FOLLOWED_DIRECTION)

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
    self.rewards_of_episode = 0

    if self.episode % 250 == 0 and self.episode != 0:
        self.logger.info(f"Saving Q-Table at episode: {self.episode}")
        np.save(f"q_table-{self.timestamp}", self.q_table)

    self.episode += 1
    self.exploration_rate = self.exploration_rate_end + (
        self.exploration_rate_initial - self.exploration_rate_end
    ) * np.exp(
        -self.exploration_decay_rate * self.episode
    )  # decay

    self.logger.info(f"Fraction of unseen states: {fraction_of_unseen_states(self.q_table)}")
    self.logger.info(f"Average seen actions per state: {avg_seen_actions(self.q_table)}")
    self.logger.info(
        f"Distribution of actions over all states: {distribution_of_best_actions(self.q_table)}"
    )


def reward_from_events(self, events: List[str]) -> int:
    """
    Returns a summed up reward/penalty for a given list of events that happened.
    """

    game_rewards = {
        e.BOMB_DROPPED: 10,
        e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 50,
        e.COIN_FOUND: 5,
        e.WAITED: -3,
        e.CRATE_DESTROYED: 4,
        e.GOT_KILLED: -50,
        e.KILLED_OPPONENT: 200,
        e.KILLED_SELF: -10,
        e.OPPONENT_ELIMINATED: 0.05,
        e.SURVIVED_ROUND: 0,
        e.INVALID_ACTION: -10,
        WAS_BLOCKED: -20,
        MOVED: -0.1,
        INCREASED_SURROUNDING_CRATES: 1.5,
        DECREASED_SURROUNDING_CRATES: -1.6,
        FOLLOWED_DIRECTION: 5,
        NOT_FOLLOWED_DIRECTION: -6,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
