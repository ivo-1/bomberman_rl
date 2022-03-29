import numpy as np
import torch

from agent_code.coli_agent_offline.decision_transformer.models.decision_transformer import (
    DecisionTransformer,
)

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    self.old_score = 0
    self.timestep = 1
    self.current_round = 0

    self.logger.info("Loading model from saved state.")

    self.action_dim = len(ACTIONS)
    self.context_window = 20
    self.hidden_size = 128
    self.dropout = 0.1

    self.radius = 3  # how many fields away from the agent can the agent see (in every direction)
    self.board_size = (
        self.radius * 2 + 1
    )  # the resulting board size (width/height) around the agent
    self.state_dim = self.board_size ** 2  # the size of the board is the square of the width/height

    self.field_state_to_idx = {
        "free": 1,
        "wall": 2,
        "crate": 3,
        "coin": 4,
        "coin_in_explosion": 5,
        "explosion": 6,
        "agent_bomb_possible": 7,
        "agent_bomb_not_possible": 8,
        "agent_on_own_bomb": 9,
        "bomb": 9,
    }

    self.scale = (
        2.5  # how much the rewards are scaled (divided by) s.t. they fall into range [0, 10]
    )

    # TODO: make this dynamic
    self.state_mean = torch.tensor(1.8590130975728476).to(device=self.device)
    self.state_std = torch.tensor(1.3812701333301785).to(device=self.device)

    self.model = DecisionTransformer(
        state_dim=self.state_dim,  # how many entries the feature vector has
        action_dim=self.action_dim,  # how many actions one can take
        max_length=self.context_window,  # context window
        max_ep_len=401,  # game of bomberman lasts max. 401 steps
        hidden_size=self.hidden_size,  # size of positional embeddings
        n_layer=3,  # GPT2
        n_head=1,  # GPT2
        n_inner=4 * self.hidden_size,  # GPT2 4 because we have 4 heads (r, s, a, t)
        activation_function="relu",  # GPT2
        n_positions=1024,  # GPT2 ("the maximum sequence length that this model might ever be used with")
        resid_pdrop=self.dropout,  # GPT2
        attn_pdrop=self.dropout,  # GPT2
    )

    path = "/Users/ivo/Studium/fml/bomberman_rl/agent_code/coli_agent_offline/decision_transformer/checkpoints/2022-03-27T14:15:36/iter_20.pt"

    self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    self.model.eval()  # evaluation (inference) mode
    self.device = "cpu"
    self.model.to(device=self.device)

    self.target_return = torch.tensor(
        9 / self.scale, device=self.device, dtype=torch.float32
    ).reshape(1, 1)

    # keep track of states, actions, rewards, timesteps and initialize with zeros/empty
    self.states = torch.zeros((1, self.state_dim), device=self.device)  # zero-state
    self.actions = torch.zeros((0, self.action_dim), device=self.device)  # empty
    self.rewards = torch.zeros(0, device=self.device)  # empty
    self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(
        1, 1
    )  # zero-timestep


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    if game_state["round"] != self.current_round:
        self.timestep = 1
        self.current_round = game_state["round"]

    # get the reward from the previous action
    previous_reward = game_state["self"][1] - self.old_score
    self.old_score = previous_reward

    current_return_to_go = self.target_return[0, -1] - (previous_reward / self.scale)

    self.target_return = torch.cat([self.target_return, current_return_to_go.reshape(1, 1)])

    # and add it to previous seen rewards and overwrite the padding "fake" reward
    self.rewards = torch.cat(
        [self.rewards[:-1], torch.tensor(previous_reward, device=self.device).unsqueeze(0)]
    )

    # The Decision Transformer always takes in (s, a, r). However, of course, at
    # this point there are no actions (and hence no rewards) because that's the whole
    # point of what should be predicted.
    #
    # We can't just pass in the current state to it. Instead we pass in the current state
    # *and* "fake" actions and rewards that are just zeros
    self.actions = torch.cat(
        [self.actions, torch.zeros((1, self.action_dim), device=self.device)], dim=0
    )
    self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)])

    self.logger.debug("Converting game state to feature vector")
    # get the current state
    state = (
        torch.from_numpy(state_to_features(self, game_state))
        .to(device=self.device)
        .reshape(1, self.state_dim)
    )

    # concatenate it with all previous seen states
    self.states = torch.cat([self.states, state], dim=0)

    self.logger.debug("Querying model for action...")
    # print(f"self timestep: {self.timestep}")
    action = self.model.get_action(
        states=(self.states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
        actions=self.actions.to(dtype=torch.float32),
        returns_to_go=self.target_return.to(dtype=torch.float32),
        timesteps=self.timesteps.to(dtype=torch.long),
    )

    # overwrite the "fake" action so that it will be used in the next timestep
    self.actions[-1] = action
    action = (
        action.detach().cpu().numpy()
    )  # we need to detach it from the device and return a numpy copy to the CPU

    # update timestep for next action
    self.timesteps = torch.cat(
        [self.timesteps, torch.ones((1, 1), device=self.device) * (self.timestep)], dim=1
    )

    self.timestep += 1

    return ACTIONS[np.argmax(action)]


def state_to_features(self, game_state: dict) -> np.array:
    """Parses game state to self.state_dim-dimensional feature vector.

    Out of bound coordinates are represented as walls (feature value 2).

    values in the vector mean the following:
        1: free
        2: wall
        3: crate
        4: coin
        5: coin which is in an explosion
        6: explosion (without coin on it)
        7: agent that can place a bomb
        8: agent that can't place a bomb
        9: agent that's standing on his own bomb
        10: bomb (without agent standing on it)
    """
    feature_vector = np.zeros(self.state_dim, dtype=int)

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

    for index, coordinate in enumerate(grid):
        # NOTE: Order is important, because in game_state["field"] neither bombs nor opponents etc. are considered
        if (
            coordinate[0] < 0 or coordinate[1] < 0 or coordinate[0] > 16 or coordinate[1] > 16
        ):  # out of bound coordinates
            feature_vector[index] = self.field_state_to_idx["wall"]
            continue

        elif (
            list(coordinate) in np.argwhere(game_state["field"] == -1).tolist()
        ):  # it's a wall (unambiguos)
            feature_vector[index] = self.field_state_to_idx["wall"]
            continue

        elif (
            list(coordinate) in np.argwhere(game_state["field"] == 1).tolist()
        ):  # it's a crate (unambiguos)
            feature_vector[index] = self.field_state_to_idx["crate"]
            continue

        # can't just check game_state["field"] == 0 (free) because it might not be free,
        # e.g. there might be an agent/a bomb/... (ambiguos). Hence stary by checking
        # if there's an agent on the field
        elif coordinate in [
            other_agent[-1] for other_agent in game_state["others"]
        ]:  # another agent is on the field

            # get the information of the agent that is on the field
            agent_on_that_field = [
                other_agents
                for other_agents in game_state["others"]
                if other_agents[-1] == coordinate
            ][0]

            if agent_on_that_field[-1] in [
                bomb[0] for bomb in game_state["bombs"]
            ]:  # standing on its own bomb
                feature_vector[index] = self.field_state_to_idx["agent_on_own_bomb"]
                continue
            elif agent_on_that_field[2]:  # agent can place bomb
                feature_vector[index] = self.field_state_to_idx["agent_bomb_possible"]
                continue
            # can only be an agent that can't place a bomb but still check
            elif not agent_on_that_field[2]:
                feature_vector[index] = self.field_state_to_idx["agent_bomb_not_possible"]
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
                feature_vector[index] = self.field_state_to_idx["coin_in_explosion"]
                continue

            else:
                raise ValueError("Impossible coin field")

        # check if there's an explosion
        elif list(coordinate) in np.argwhere(game_state["explosion_map"] != 0).tolist():
            feature_vector[index] = self.field_state_to_idx["explosion"]
            continue

        # can only be free but nonetheless check
        elif list(coordinate) in np.argwhere(game_state["explosion_map"] == 0).tolist():
            feature_vector[index] = self.field_state_to_idx["free"]
            continue

        else:
            raise ValueError("Unknown field state!")
    return feature_vector
