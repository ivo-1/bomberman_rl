import os
import pickle
import random

import numpy as np
import torch
from decision_transformer.models.decision_transformer import DecisionTransformer

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.logger.info("Loading model from saved state.")
    self.model = DecisionTransformer(
        state_dim=49,  # how many entries the feature vector has (7*7=49)
        action_dim=6,  # how many actions one can take
        max_length=20,  # context window
        max_ep_len=401,  # game of bomberman lasts max. 401 steps
        hidden_size=128,  # size of positional embeddings
        n_layer=3,  # GPT2
        n_head=1,  # GPT2
        n_inner=4 * 128,  # GPT2
        activation_function="relu",  # GPT2
        n_positions=1024,  # GPT2 ("the maximum sequence length that this model might ever be used with")
        resid_pdrop=0.1,  # GPT2
        attn_pdrop=0.1,  # GPT2
    )
    self.model.load_state_dict(torch.load("our/path/to/state_dict"))
    self.model.eval()  # evaluation (inference) mode
    self.device = "cpu"
    self.model.to(device=self.device)

    self.scale = (
        2.5  # how much the rewards are scaled (divided by) s.t. they fall into range [0, 10]
    )
    self.target_return = torch.tensor(15 / self.scale, device=self.device).reshape(1, 1)
    self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.logger.debug("Querying model for action.")

    return


def state_to_features(self, game_state: dict) -> np.array:
    """Parses game state to 49-dimensional feature vector

    Out of board coordinates are represented as walls (feature value 1).

    field to value:
        1: free
        2: wall
        3: crate
        4: coin
        5: coin which is in an explosion
        6: explosion (without coin on it)
        7: agent that can place a bomb
        8: agent that can't place a bomb
        9: agent that's standing on a bomb (i.e. his own bomb)
        10: bomb (without agent standing on it)
    """
    feature_vector = np.zeros(
        self.board_size ** 2, dtype=int
    )  # -128 to 127 suffices and saves space

    own_position = game_state["self"][-1]

    # get the 7x7 board around ourselves (3 in every direction)
    x_min = own_position[0] - self.radius
    y_min = own_position[1] - self.radius

    x_max = own_position[0] + self.radius
    y_max = own_position[1] + self.radius

    grid = []
    for x in np.arange(x_min, x_max + 1):
        for y in np.arange(y_min, y_max + 1):
            grid.append((x, y))

    assert len(grid) == self.board_size ** 2

    for index, coordinate in enumerate(grid):
        # print(index)
        # print(coordinate)
        # NOTE: Order is important, because in game_state["field"] neither bombs nor opponents etc. are considered

        if (
            coordinate[0] < 0 or coordinate[1] < 0 or coordinate[0] > 16 or coordinate[1] > 16
        ):  # out of bound coordinates
            # print("Out of bound")
            feature_vector[index] = self.field_state_to_idx["wall"]
            continue

        elif (
            list(coordinate) in np.argwhere(game_state["field"] == -1).tolist()
        ):  # it's a wall (unambiguos)
            # print(np.argwhere(game_state["field"]==-1))
            # print(coordinate)
            # break
            feature_vector[index] = self.field_state_to_idx["wall"]
            continue

        elif (
            list(coordinate) in np.argwhere(game_state["field"] == 1).tolist()
        ):  # it's a crate (unambiguos)
            feature_vector[index] = self.field_state_to_idx["crate"]
            continue

        # can't check game_state["field"] == 0 (free) because it might not be free (ambiguos)
        # check if there's an agent on the field
        elif coordinate in [
            other_agent[-1] for other_agent in game_state["others"]
        ]:  # another agent
            # print("There's an agent on this field!")
            # print(game_state["others"])

            # investigating the agent further:
            agent_on_that_field = [
                other_agents
                for other_agents in game_state["others"]
                if other_agents[-1] == coordinate
            ][0]
            # print(agent_on_that_field)

            if agent_on_that_field[-1] in [
                bomb[0] for bomb in game_state["bombs"]
            ]:  # standing on its own bomb
                feature_vector[index] = self.field_state_to_idx["agent_on_own_bomb"]
                # print("AGENT ON OWN BOMB")
                # print([other_agent[-1] for other_agent in game_state["others"] if other_agent[0][2]])
                continue
            elif agent_on_that_field[2]:  # agent can place bomb
                feature_vector[index] = self.field_state_to_idx["agent_bomb_possible"]
                # print("AGENT CAN PLACE BOMB")
                continue
            # can only be an agent that can't place a bomb but still check
            elif not agent_on_that_field[2]:
                feature_vector[index] = self.field_state_to_idx["agent_bomb_not_possible"]
                # print("AGENT CANNOT PLACE BOMB")
                continue
            else:
                raise ValueError("Impossible agent field!")

        # check if there's a bomb
        elif coordinate in [bomb[0] for bomb in game_state["bombs"]]:
            feature_vector[index] = self.field_state_to_idx["bomb"]
            continue

        # check if there's a coin
        elif coordinate in game_state["coins"]:
            if (
                game_state["explosion_map"][coordinate[0]][coordinate[1]] == 0
            ):  # the coin is not in an explosion
                feature_vector[index] = self.field_state_to_idx["coin"]
                continue

            elif (
                game_state["explosion_map"][coordinate[0]][coordinate[1]] == 1
            ):  # the coin is in an explosion
                # print("EXPLOSION COIN")
                feature_vector[index] = self.field_state_to_idx["coin_in_explosion"]
                continue

            else:
                raise ValueError("Impossible coin field")

        # check if there's an explosion
        elif (
            list(coordinate) in np.argwhere(game_state["explosion_map"] != 0).tolist()
        ):  # it's an explosion
            feature_vector[index] = self.field_state_to_idx["explosion"]
            continue

        # can only be free but nonetheless check
        elif (
            list(coordinate) in np.argwhere(game_state["explosion_map"] == 0).tolist()
        ):  # it's free
            feature_vector[index] = self.field_state_to_idx["free"]
            continue

        else:
            print(index)
            print(coordinate)
            raise ValueError("Unknown field state!")

    # print(feature_vector)

    # MAYBE: encode the fact if we can drop a bomb into feature vector
    # feature_vector.append(game_state["self"][2])
    return feature_vector
