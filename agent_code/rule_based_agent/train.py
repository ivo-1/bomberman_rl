from datetime import datetime
from typing import List

import numpy as np

import events as e

from .callbacks import state_to_features

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def _one_hot_encode(x) -> np.array:
    encoded = np.zeros(len(ACTIONS), dtype=int)
    encoded[x] = 1
    return encoded


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s') # TODO: we might need s' but I assume we don't
    self.episode_trajectory = []
    self.trajectories_over_episodes = []
    self.episode = 1  # need to keep track of episodes
    self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S:%f")


def game_events_occurred(
    self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    if self_action is None:
        self.logger.debug("Only happens when invalid action was chosen")
        self_action = "WAIT"  # this is the result of an invalid action

    # state_to_features is defined in callbacks.py
    self.episode_trajectory.append(
        np.array(
            [
                state_to_features(self, old_game_state),
                _one_hot_encode(ACTIONS.index(self_action)),
                reward_from_events(self, events),
            ],
            dtype=object,
        )  # state_to_features(self, new_game_state), new game state shouldn't be necessary
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if last_action is None:
        self.logger.debug("Only happens when invalid action was chosen")
        last_action = "WAIT"  # this is the result of an invalid action

    self.episode_trajectory.append(
        np.array(
            [
                state_to_features(self, last_game_state),
                _one_hot_encode(ACTIONS.index(last_action)),
                reward_from_events(self, events),
            ],
            dtype=object,
        )
    )

    # store the episode trajectory into the global variable
    self.trajectories_over_episodes.append(np.array(self.episode_trajectory, dtype=object))

    # print(f"Episode: {self.episode}")

    # Store all encountered trajectories of (state, action, reward) in a npy when last episode finished
    if self.n_rounds == self.episode:  # we're in the last planned episode
        trajectories = np.array(self.trajectories_over_episodes, dtype=object)
        print("How many trajectories?\n")
        print(len(trajectories))
        print("Average trajectory length:\n")
        print(np.average([len(trajectory) for trajectory in trajectories]))

        np.save(
            f"../coli_agent_offline/trajectories/trajectories_{self.timestamp}",
            trajectories,
        )

    self.episode_trajectory = []  # reset episode trajectory
    self.episode += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {e.COIN_COLLECTED: 1, e.KILLED_OPPONENT: 5}
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
