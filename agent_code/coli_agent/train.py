from typing import List

import numpy as np

import events as e


def setup_training(self):
    """Sets up training"""
    self.number_of_episodes = 100
    self.exploration_rate = self.exploration_rate_initial


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called once after each time step except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.
    """


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent (?) after the last step of a round."""
    pass


def reward_from_events(self, events: List[str]) -> int:
    """
    Returns a summed up reward/penalty for a given list of events that happened

    Currently not assigning penalty to INVALID_ACTION because then it shouldn't even get this far (?).
    Also not assigning reward/penalty to definitely(?) neutral actions MOVE LEFT/RIGHT/UP/DOWN or WAIT.
    """
    # TODO: customs events
    # TODO: different rewards for different learning subtasks?
    game_rewards = {
        e.BOMB_DROPPED: 0.25,  # adjust aggressiveness
        # e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 1,
        e.COIN_FOUND: 0.5,
        # e.CRATE_DESTROYED: 0,  # possibly use if agent isn't destroying enough crates
        e.GOT_KILLED: -5,  # adjust passivity
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,  # you dummy
        # e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 1,  # could possibly lead to not being active
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def training_loop(self):
    for episode in range(self.number_of_episodes):
        self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end
        ) * np.exp(-self.exploration_decay_rate * episode)
