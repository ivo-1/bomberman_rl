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
from agent_code.coli_agent.callbacks import (
    ACTIONS,
    get_neighboring_tiles_until_wall,
    state_to_features,
)
from agent_code.coli_agent.plots import get_plots

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions

# --- Custom Events ---

# - Events with direct state feature correspondence -

WAS_BLOCKED = "WAS_BLOCKED"  # tried to move into a wall/crate/enemy/explosion (strong penalty)
MOVED = "MOVED"  # moved somewhere (and wasn't blocked)

PROGRESSED = "PROGRESSED"  # in last 5 turns, agent visited at least 3 unique tiles
STAGNATED = "STAGNATED"  # opposite for stronger effect

FLED = "FLED"  # was in "danger zone" of a bomb and moved out of it (reward)
SUICIDAL = (
    "SUICIDAL"  # moved from safe field into "danger" zone of bomb (penalty, higher than reward)
)

DECREASED_DISTANCE = (
    "DECREASED_DISTANCE"  # decreased length of shortest path to nearest coin or crate BY ONE
)
INCREASED_DISTANCE = (
    "INCREASED_DISTANCE"  # increased length of shortest path to nearest coin or crate BY ONE
)

FOLLOWED_DIRECTION = "FOLLOWED_DIRECTION"  # went in direction indicated by coin/crate feature
NOT_FOLLOWED_DIRECTION = "NOT_FOLLOWED_DIRECTION"

INCREASED_SURROUNDING_CRATES = (
    "INCREASED_SURROUNDING_CRATES"  # increased or stayed the same; low reward
)
DECREASED_SURROUNDING_CRATES = (
    "DECREASED_SURROUNDING_CRATES"  # equal or slightly higher penalty for balance
)

# - Events without direct state feature correspondence -

# idea: Agent-Coin ratio: reward going after crates when there's many coins left (max: 9) and
# reward going after agents when there aren't
# idea: reward caging enemies

# more fine-grained bomb area movements: reward/penalize moving one step away from/towards bomb,
# when agent is already in "danger zone"
INCREASED_BOMB_DISTANCE = "INCREASED_BOMB_DISTANCE"  # increased or stayed the same
DECREASED_BOMB_DISTANCE = "DECREASED_BOMB_DISTANCE"


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

    if old_feature_dict["bomb_danger_zone"] == 1 and new_feature_dict["bomb_danger_zone"] == 0:
        events.append(FLED)
    elif old_feature_dict["bomb_danger_zone"] == 0 and new_feature_dict["bomb_danger_zone"] == 1:
        events.append(SUICIDAL)

    if new_feature_dict["progressed"] == 1 and not e.INVALID_ACTION in events:
        events.append(PROGRESSED)
    if new_feature_dict["progressed"] == 0:
        events.append(STAGNATED)

    if new_game_state["self"][-1] != old_game_state["self"][-1]:
        events.append(MOVED)

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

    crate_counter = [0, 0]  # [old, new]
    for tile in old_neighbors:
        if old_game_state["field"][tile[0]][tile[1]] == 1:
            crate_counter[0] += 1
    for tile in new_neighbors:
        if new_game_state["field"][tile[0]][tile[1]] == 1:
            crate_counter[1] += 1
    if crate_counter[0] < crate_counter[1]:
        events.append(INCREASED_SURROUNDING_CRATES)
    elif crate_counter[0] > crate_counter[1]:
        events.append(DECREASED_SURROUNDING_CRATES)

    if old_feature_dict["bomb_danger_zone"] == 1:
        bomb_positions = []
        for tile in old_neighbors:
            if tile in [bomb[0] for bomb in old_game_state["bombs"]]:
                bomb_positions.append(tile)
        shortest_old_distance = 1000
        for bp in bomb_positions:
            distance = abs(
                sum(np.array(old_game_state["self"][-1]) - np.array(bp))
            )  # e.g.: [13,8] - [13,10] = [0,-2] -> |-2|
            if distance < shortest_old_distance:
                shortest_old_distance = distance
        for tile in new_neighbors:
            if tile in [bomb[0] for bomb in new_game_state["bombs"]]:
                bomb_positions.append(tile)
        shortest_new_distance = 1000
        for bp in bomb_positions:
            distance = abs(sum(np.array(new_game_state["self"][-1]) - np.array(bp)))
            if distance < shortest_new_distance:
                shortest_new_distance = distance
        if shortest_new_distance < shortest_old_distance:
            events.append(DECREASED_BOMB_DISTANCE)
        elif shortest_new_distance >= shortest_old_distance:
            events.append(INCREASED_BOMB_DISTANCE)

    if self.previous_distance < self.current_distance:
        events.append(DECREASED_DISTANCE)
    elif self.previous_distance >= self.current_distance:
        events.append(INCREASED_DISTANCE)

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

    if self.episode % 9 == 0 and self.episode != 0:
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
        # e.BOMB_DROPPED: 10,  # adjust aggressiveness
        # # e.BOMB_EXPLODED: 0,
        # e.COIN_COLLECTED: 50,
        # # e.COIN_FOUND: 5,  # direct consequence from crate destroyed, redundant reward?
        # e.WAITED: -3,  # adjust passivity
        # e.CRATE_DESTROYED: 4,
        # e.GOT_KILLED: -50,  # adjust passivity
        # e.KILLED_OPPONENT: 200,
        # e.KILLED_SELF: -10,  # you dummy --- this *also* triggers GOT_KILLED
        # e.OPPONENT_ELIMINATED: 0.05,  # good because less danger or bad because other agent scored points?
        # # e.SURVIVED_ROUND: 0,  # could possibly lead to not being active - actually penalize if agent too passive?
        # # necessary? (maybe for penalizing trying to move through walls/crates) - yes, seems to be necessary to
        # # learn that one cannot place a bomb after another placed bomb is still not exploded
        # e.INVALID_ACTION: -10,
        # WAS_BLOCKED: -20,
        # MOVED: -0.1,
        # PROGRESSED: 10,  # higher?
        # STAGNATED: -3,  # higher? lower?
        # FLED: 15,
        # SUICIDAL: -15,
        # DECREASED_DISTANCE: 8,
        # INCREASED_DISTANCE: -8.1,  # higher? lower? idk
        # INCREASED_SURROUNDING_CRATES: 1.5,
        # DECREASED_SURROUNDING_CRATES: -1.6,
        # INCREASED_BOMB_DISTANCE: 5,
        # DECREASED_BOMB_DISTANCE: -5.1,
        FOLLOWED_DIRECTION: 5,  # possibly create penalty
        NOT_FOLLOWED_DIRECTION: -6,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
