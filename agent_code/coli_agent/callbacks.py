import glob
import itertools
import os
from collections import deque
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np
from sklearn import neighbors
from sympy import exp, solve, symbols

from settings import COLS, ROWS

Coordinate = Tuple[int]

Graph = nx.Graph
Action = str

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
SHORTEST_PATH_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


def setup(self):
    """Sets up everything. (First call)"""

    self.new_state = None
    self.history = deque(maxlen=5)  # tiles visited
    self.lattice_graph = nx.grid_2d_graph(m=COLS, n=ROWS)
    self.previous_distance = 0
    self.current_distance = 0
    self.state_list = list_possible_states()

    # for plotting
    self.q_table_fraction_unseen = []
    self.q_table_average_seen = []
    self.q_table_distribution_of_actions = []

    self.exploration_rates_of_episodes = []
    self.rewards_of_episodes = []
    self.game_scores_of_episodes = []

    # find latest q_table
    list_of_q_tables = glob.glob(
        "./q_tables/*.npy"
    )  # * means all if need specific format then *.csv
    self.latest_q_table_path = max(list_of_q_tables, key=os.path.getctime)
    # self.latest_q_table_path = "q_table-2022-03-14T162802-node45.npy"
    self.latest_q_table = np.load(self.latest_q_table_path)

    self.logger.info(f"Using q-table: {self.latest_q_table_path}")

    # train if flag is present or if there is no q_table present
    if self.train or not os.path.isfile(self.latest_q_table_path):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = 240  # TODO: make this dynamic

        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05  # at end of all episodes

        self.exploration_decay_rate = _determine_exploration_decay_rate(self)

        if self.continue_training:
            self.logger.info("Continuing training on latest q_table")
            self.q_table = self.latest_q_table

        else:
            self.logger.info("Starting training from scratch")
            self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))

        # Finally this will call setup_training in train.py

    else:
        self.logger.info("Using latest Q-Table for testing")
        self.q_table = self.latest_q_table


def list_possible_states() -> np.array:
    states = list(
        itertools.product(
            (
                "DOWN",
                "UP",
                "RIGHT",
                "LEFT",
            ),  # coin direction  - kind of a "fall back" when there's nothing else to do
            (
                "CLEAR",
                "DOWN",
                "UP",
                "RIGHT",
                "LEFT",
            ),  # bomb safety direction  - highest priority in rewards
            (0, 1),  # in enemy zone
            (
                0,
                1,
            ),  # is it safe to drop a bomb? - if in enemy zone and safe to bomb ... if many crates and safe to bomb ...
            (
                "ZERO",
                "FEW",
                "MANY",
            )  # how many surrounding crates are there (and how close are they?) - high bomb dropping reward when this is high
            # enemy trapped, self trapped/quagmire, which actions are possible, enemy attack + flight directions, attack or flee?, mode agent/coin ratio, coin path long or short, safe to go direction
        )
    )
    state_dicts = []
    for vector in states:
        state_dict = {
            "coin_direction": vector[0],
            "bomb_safe_direction": vector[1],
            "in_enemy_zone": vector[2],
            "safe_to_bomb": vector[3],
            "surrounding_crates": vector[4],
        }
        state_dicts.append(state_dict)
    print(state_dicts[0])
    return state_dicts


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    if self.new_state is None:  # is always None in test case
        self.old_state = state_to_features(self, game_state)
    else:  # in train case this is set in game_events_occured
        self.old_state = self.new_state
    state = self.old_state

    # only for logging
    safe_coins = [
        coin
        for coin in game_state["coins"]
        if coin
        not in [index for index, field in np.ndenumerate(game_state["explosion_map"]) if field != 0]
    ]
    self.logger.info(f"Current safe coins: {safe_coins}")
    self.logger.info(f"Current self coord: {game_state['self'][-1]}")

    if self.train and np.random.random() < self.exploration_rate:
        self.logger.info("Exploring")
        action = np.random.choice(ACTIONS)
        self.logger.info(f"Action chosen: {action}")
        return action

    self.logger.info("Exploiting")
    # TODO: Do we want to go 100% exploitation once we have learnt the q-table?
    # Alternative is to sample from the learnt q_table distribution.
    self.logger.debug(f"State: {state}")
    action = ACTIONS[np.argmax(self.q_table[state])]
    self.logger.info(f"Action chosen: {action}")
    return action


def _determine_exploration_decay_rate(self) -> float:
    """Determines the appropriate decay rate s.t. self.exploration_rate_end (approximately) is
    reached after self.n_rounds."""
    x = symbols("x", real=True)
    expr = (
        self.exploration_rate_end
        + (self.exploration_rate_initial - self.exploration_rate_end) * exp(-x * self.n_rounds)
    ) - (self.exploration_rate_end + 0.005)
    solution = solve(expr, x)[0]
    self.logger.info(f"Determined exploration decay rate: {solution}")
    return float(solution)


def _get_neighboring_tiles(own_coord, n) -> List[Coordinate]:
    own_coord_x = own_coord[0]
    own_coord_y = own_coord[1]
    neighboring_coordinates = []
    for i in range(1, n + 1):
        neighboring_coordinates.append((own_coord_x, own_coord_y + i))  # down in the matrix
        neighboring_coordinates.append((own_coord_x, own_coord_y - i))  # up in the matrix
        neighboring_coordinates.append((own_coord_x + i, own_coord_y))  # right in the matrix
        neighboring_coordinates.append((own_coord_x - i, own_coord_y))  # left in the matrix
    return neighboring_coordinates


def get_neighboring_tiles_until_wall(own_coord, n, game_state) -> List[Coordinate]:
    directions = ["N", "E", "S", "W"]
    own_coord_x, own_coord_y = own_coord[0], own_coord[1]
    all_good_fields = []

    for d, _ in enumerate(directions):
        good_fields = []
        for i in range(1, n + 1):
            try:
                if directions[d] == "N":
                    if (
                        game_state["field"][own_coord_x][own_coord_y + i] == 0
                        or game_state["field"][own_coord_x][own_coord_y + i] == 1
                    ):
                        good_fields += [(own_coord_x, own_coord_y + i)]
                    else:
                        break
                elif directions[d] == "E":
                    if (
                        game_state["field"][own_coord_x + i][own_coord_y] == 0
                        or game_state["field"][own_coord_x + i][own_coord_y] == 1
                    ):
                        good_fields += [(own_coord_x + i, own_coord_y)]
                    else:
                        break
                elif directions[d] == "S":
                    if (
                        game_state["field"][own_coord_x][own_coord_y - i] == 0
                        or game_state["field"][own_coord_x][own_coord_y - i] == 1
                    ):
                        good_fields += [(own_coord_x, own_coord_y - i)]
                    else:
                        break
                elif directions[d] == "W":
                    if (
                        game_state["field"][own_coord_x - i][own_coord_y] == 0
                        or game_state["field"][own_coord_x - i][own_coord_y] == 1
                    ):
                        good_fields += [(own_coord_x - i, own_coord_y)]
                    else:
                        break
            except IndexError:
                # print("Border")
                break

        all_good_fields += good_fields

    return all_good_fields


def _get_graph(self, game_state, crates_as_obstacles=True) -> Graph:
    """Calculates the adjacency matrix of the current game state.
    Every coordinate is a node.]

    Vertex between nodes <==> both nodes are empty

    Considers walls, crates, active explosions and (maybe other players) as "walls", i.e. not connected"""

    if crates_as_obstacles:
        # walls and crates are obstacles
        obstacles = [index for index, field in np.ndenumerate(game_state["field"]) if field != 0]

    else:
        # only walls are obstacles
        obstacles = [index for index, field in np.ndenumerate(game_state["field"]) if field == -1]

    # TODO: Find out what works better - considering other players as obstacles (technically true) or not
    # for other_player in game_state["others"]:
    # obstacles.append(other_player[3])  # third element stores the coordinates

    active_explosions = [
        index for index, field in np.ndenumerate(game_state["explosion_map"]) if field != 0
    ]

    bombs = [
        coordinate
        for coordinate, _ in game_state["bombs"]
        if coordinate != game_state["self"][-1]
        and coordinate not in [other_agent[-1] for other_agent in game_state["others"]]
    ]

    # self.logger.debug(f"Active explosions: {active_explosions}")
    # print(f"Active explosion: {active_explosions}")
    # print(f"Bombs: {game_state['bombs']}")
    obstacles += active_explosions
    obstacles += bombs

    self.logger.debug(f"Obstacles: {obstacles}")

    graph = nx.grid_2d_graph(m=COLS, n=ROWS)

    # inplace operation
    graph.remove_nodes_from(obstacles)  # removes nodes and all edges of that node
    return graph


def _find_shortest_path(graph, a, b) -> Tuple[Graph, int]:
    """Calclulates length of shortest path at current time step (without looking ahead to the future)
    between points a and b."""
    shortest_path = None
    # use Djikstra to find shortest path
    try:
        shortest_path = nx.shortest_path(graph, source=a, target=b, weight=None, method="dijkstra")
    except nx.exception.NodeNotFound as e:
        print(graph.nodes)
        raise e

    shortest_path_length = len(shortest_path) - 1  # because path considers self as part of the path
    return shortest_path, shortest_path_length


def _get_action(self, self_coord, shortest_path) -> Action:
    goal_coord = shortest_path[1]  # 0th element is self_coord

    self.previous_distance = self.current_distance
    self.current_distance = len(shortest_path) - 1
    self.logger.debug(f"self.previous_distance is {self.previous_distance}")
    self.logger.debug(f"self.current_distance is {self.current_distance}")
    self.logger.info(f"Determined goal at {goal_coord} from shortest path feature")

    # x-coord is the same
    if self_coord[0] == goal_coord[0]:
        if self_coord[1] + 1 == goal_coord[1]:
            return "DOWN"

        elif self_coord[1] - 1 == goal_coord[1]:
            return "UP"

    # y-coord is the same
    elif self_coord[1] == goal_coord[1]:
        if self_coord[0] + 1 == goal_coord[0]:
            return "RIGHT"

        elif self_coord[0] - 1 == goal_coord[0]:
            return "LEFT"


def _shortest_path_feature(self, game_state) -> Action:
    """
    Computes the direction along the shortest path as follows:

    If no coins and no crates exist --> random

    If no coins but a crate exists --> towards nearest crate

    If coins:

        if no coin path possible:
            towards nearest coin (thus towards first crate that's in the way)

        elif exactly one coin path possible:
            # even though there might be a coin that's much closer but
            # blocked or someone else is closer
            towards nearest coin

        elif more than one coin path possible:
            try:
                towards nearest coin that no one else is more near to

            except there is no coin that our agent is nearest to:
                towards nearest coin
    """
    graph = _get_graph(self, game_state)
    graph_with_crates = _get_graph(self, game_state, crates_as_obstacles=False)

    self.logger.debug(f"Current Graph nodes: {graph.nodes}")
    self_coord = game_state["self"][3]

    safe_coins = [
        coin
        for coin in game_state["coins"]
        if coin
        not in [index for index, field in np.ndenumerate(game_state["explosion_map"]) if field != 0]
    ]

    # no coins on board and no crates (maybe also no opponents ==> suicide?) ==> just return something
    if not any(safe_coins) and not any(
        [index for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    ):
        return np.random.choice(SHORTEST_PATH_ACTIONS)

    elif not any(safe_coins):
        best = (None, np.inf)

        crates_coordinates = [
            index for index, field in np.ndenumerate(game_state["field"]) if field == 1
        ]

        # self.logger.debug(f"Crates coordinates: {crates_coordinates}")

        for crate_coord in crates_coordinates:
            try:
                current_path, current_path_length = _find_shortest_path(
                    graph_with_crates, self_coord, crate_coord
                )

            # in some edge cases it can happen that a crate is unreachable b/c of explosion
            # even though we don't consider crates themselves as obstacles
            except nx.exception.NetworkXNoPath:
                self.logger.info("Crazy edge case (unreachable crate) occured!")
                continue

            # self.logger.debug(f"Current path: {current_path} with path length: {current_path_length}
            # to crate at {crate_coord}")

            # not gonna get better than 1, might save a bit of computation time
            if current_path_length == 1:
                self.logger.info("Standing directly next to crate!")
                return _get_action(self, self_coord, current_path)

            elif current_path_length < best[1]:
                best = (current_path, current_path_length)

        if best == (None, np.inf):
            self.logger.info(
                "There are no coins and no crate is reachable even if not considering crates as obstacles"
            )
            return np.random.choice(SHORTEST_PATH_ACTIONS)

        return _get_action(self, self_coord, best[0])

    # there is a coin
    else:
        self.logger.info("There is a safe coin and it is not *in* an explosion")
        shortest_paths_to_coins = []

        # find shortest paths to all coins by all agents
        for coin_coord in safe_coins:
            try:
                current_path, current_path_length = _find_shortest_path(
                    graph, self_coord, coin_coord
                )
                current_reachable = True

            # coin path not existent
            except nx.exception.NetworkXNoPath:
                try:
                    current_path, current_path_length = _find_shortest_path(
                        graph_with_crates, self_coord, coin_coord
                    )
                    current_reachable = False
                except nx.exception.NetworkXNoPath:
                    self.logger.info(
                        "Crazy edge case (unreachable coin for us even though crates not \
                        considered as obstacles) occured!"
                    )
                    continue

            # all other agents are dead
            if not any(game_state["others"]):
                shortest_paths_to_coins.append(
                    (
                        (current_path, current_path_length, current_reachable),
                        (None, np.inf),
                    )
                )
                continue

            for other_agent in game_state["others"]:
                best_other_agent = (None, np.inf)
                other_agent_coord = other_agent[3]
                try:
                    (
                        current_path_other_agent,
                        current_path_length_other_agent,
                    ) = _find_shortest_path(graph, other_agent_coord, coin_coord)
                    current_other_agent_reachable = True

                # other agent can't reach coin
                except nx.exception.NetworkXNoPath:

                    try:
                        (
                            current_path_other_agent,
                            current_path_length_other_agent,
                        ) = _find_shortest_path(graph_with_crates, other_agent_coord, coin_coord)
                        current_other_agent_reachable = False

                    except nx.exception.NetworkXNoPath:
                        self.logger.info(
                            f"Crazy edge case (unreachable coin for other agent {other_agent} even \
                            though crates not considered as obstacles) occured!"
                        )
                        continue

                # penalize with heuristic of 7 more fields if unreachable
                if not current_other_agent_reachable:
                    current_path_length_other_agent += 7

                if current_path_length_other_agent < best_other_agent[1]:
                    best_other_agent = (
                        current_path_other_agent,
                        current_path_length_other_agent,
                        current_other_agent_reachable,
                    )

            shortest_paths_to_coins.append(
                (
                    (current_path, current_path_length, current_reachable),
                    best_other_agent,
                )
            )

        # this happens if none of the coins are reachable by us even if considering crates as obstacles
        if not any(shortest_paths_to_coins):
            return np.random.choice(SHORTEST_PATH_ACTIONS)

        # sort our [0] paths ascending by length [1]
        shortest_paths_to_coins.sort(key=lambda x: x[0][1])

        shortest_paths_to_coins_reachable = [
            shortest_path_to_coin[0][2] for shortest_path_to_coin in shortest_paths_to_coins
        ]

        # if none of our [0] shortest paths are actually reachable [2] we just go towards
        # the nearest one (i.e. to its nearest crate)
        if not any(shortest_paths_to_coins_reachable):
            self.logger.debug("No coin reachable ==> Going towards nearest one")
            return _get_action(
                self, self_coord, shortest_paths_to_coins[0][0][0]
            )  # shortest [0] (because sorted) that is ours [0] and the actual path [0]

        # if exactly one of our [0] shortest paths is reachable [2] we go towards that one
        elif shortest_paths_to_coins_reachable.count(True) == 1:
            self.logger.debug("Exactly one coin reachable ==> Going towards that one")
            index_of_reachable_path = shortest_paths_to_coins_reachable.index(True)
            return _get_action(
                self, self_coord, shortest_paths_to_coins[index_of_reachable_path][0][0]
            )

        # if more than one shortest path is reachable we got towards the one that we are closest
        # and reachable to and no one else being closer
        for shortest_path_to_coin in shortest_paths_to_coins:

            # we are able to reach it and we are closer
            if (
                shortest_path_to_coin[0][2] is True
                and shortest_path_to_coin[0][1] <= shortest_path_to_coin[1][1]
                and shortest_path_to_coin[0][1]
                != 0  # we are standing on a coin because we spawned on it --> correct would be to "WAIT"
                # but we want to stick to "UP", "DOWN", "LEFT" and "RIGHT" hence we just return second
                # closest coin
            ):
                self.logger.debug(
                    f"We are able to reach coin at {shortest_path_to_coin[0]} and we are closest to it"
                )
                return _get_action(self, self_coord, shortest_path_to_coin[0][0])

        self.logger.info("Fallback Action")
        # unless we are not closest to any of our reachable coins then we return action that leads us to
        # the coin we are nearest too anyway
        try:
            return _get_action(self, self_coord, shortest_paths_to_coins[0][0][0])

        # it is theoretically possible that coins are not reachable by our agent even if we don't consider
        # crates as obstacles where shortest_paths_to_coins will be empty
        except IndexError:
            return np.random.choice(SHORTEST_PATH_ACTIONS)


def enemy_zone_feature(game_state: dict) -> int:
    own_position = game_state["self"][-1]
    all_enemy_fields = []
    if_dangerous = []
    for enemy in game_state["others"]:
        neighbours_until_wall = get_neighboring_tiles_until_wall(
            enemy[-1], 3, game_state=game_state
        )
        if neighbours_until_wall:
            all_enemy_fields += neighbours_until_wall

    if len(all_enemy_fields) > 0:
        for bad_field in all_enemy_fields:
            in_danger = own_position == bad_field
            if_dangerous.append(in_danger)

        return int(any(if_dangerous))
    else:
        return 0


def surrounding_crates_feature(game_state: dict) -> int:
    own_position = game_state["self"][-1]
    neighbours = get_neighboring_tiles_until_wall(own_position, 3, game_state=game_state)
    crate_coordinates = []

    if neighbours:
        for coord in neighbours:
            if game_state["field"][coord[0]][coord[1]] == 1:
                crate_coordinates += [coord]

        if len(crate_coordinates) == 0:
            return 0
        elif 1 <= len(crate_coordinates) < 4:
            return 1
        elif len(crate_coordinates) >= 4:
            return 2

    return 0


def bomb_safety_direction_feature(self, game_state):
    """
    Issues with this feature:
    * shortest path is in this case not necessarily best path
    * shortest path might change frequently because enemy is moving away -> leads to moving back and forth
    """
    own_position = game_state["self"][-1]
    relevant_neighbors = get_neighboring_tiles_until_wall(own_position, 3, game_state)
    bomb_positions = [bomb[0] for bomb in game_state["bombs"]]
    self.logger.debug(f"Bomb positions: {bomb_positions}")

    if not any(
        [neighbor in bomb_positions for neighbor in relevant_neighbors]
    ):  # agent is not in any future explosion zone of bomb
        return "CLEAR"

    bomb_explosion_tiles = []
    reach = 4  # how far we can still go before most urgent bomb blows up
    for bomb in game_state["bombs"]:
        bomb_explostion_tiles += get_neighboring_tiles_until_wall(bomb[0], 3, game_state)
        if bomb[1] + 1 < reach:
            reach = bomb[1] + 1
    self.logger.debug(f"Reach: {reach}")
    graph = _get_graph(self, game_state)
    available_neighbors = _get_neighboring_tiles(own_position, reach)

    shortest_path = None
    shortest_distance = 1000
    for n in available_neighbors:
        if n not in graph or n in bomb_explosion_tiles:
            self.logger.debug(f"Skipping tile {n} as not available for bomb flight")
            continue
        try:
            current_shortest_path, current_shortest_distance = _find_shortest_path(
                graph, own_position, n
            )
            if current_shortest_distance < shortest_distance:
                shortest_path = current_shortest_path
                shortest_distance = current_shortest_distance
        except nx.exception.NetworkXNoPath:
            continue

    self.logger.debug(f"Bomb safety goal: {shortest_path[1]}")  # ?

    if not shortest_path:
        return "NO_WAY_OUT"

    return _get_action(self, own_position, shortest_path)


def safe_to_bomb_feature(self, game_state) -> int:
    if not game_state["self"][2]:
        return 0
    # return 0 if there would be no path away from bomb if bomb was in place of own coord
    return 1


def state_to_features(self, game_state) -> np.array:
    """Parses game state to features"""

    state_dict = {
        "coin_direction": None,
        "bomb_safe_direction": None,
        "in_enemy_zone": None,
        "safe_to_bomb": None,
        "surrounding_crates": None,
    }

    # Feature 1: Next direction in shortest path to nearest coin or crate
    direction = _shortest_path_feature(self, game_state)
    if direction in ["DOWN", "UP", "RIGHT", "LEFT"]:
        state_dict["coin_direction"] = direction
    else:
        raise ValueError(f"Invalid directon to nearest coin/crate: {direction}")

    # Feature 2: Next direction in path to bomb safety zone, or "CLEAR" if not in bomb zone
    bomb_safety_result = bomb_safety_direction_feature(self, game_state)
    if bomb_safety_result in ["DOWN", "UP", "RIGHT", "LEFT", "CLEAR"]:
        state_dict["coin_direction"] = bomb_safety_result
    elif bomb_safety_result == "NO_WAY_OUT":
        state_dict["coin_direction"] = np.random.choice([SHORTEST_PATH_ACTIONS])
    else:
        raise ValueError(f"Invalid directon to bomb safety: {bomb_safety_result}")

    # Feature 3: Is agent within bombing reach of other agent?
    state_dict["enemy_danger_zone"] = enemy_zone_feature(game_state=game_state)

    # Feature 4: Is it safe to plant a bomb?
    answer = safe_to_bomb_feature(self, game_state)
    if answer:
        state_dict["safe_to_bomb"] = answer
    else:
        raise ValueError("'Safe to bomb' is None")

    # Feature 8: amount of crates within destruction reach: ZERO, FEW or MANY
    state_dict["surrounding_crates"] = surrounding_crates_feature(game_state=game_state)

    self.logger.info(f"Feature dict: {state_dict}")

    for i, state in enumerate(self.state_list):
        if state == state_dict:
            return i

    raise ReferenceError(
        f"State dict created by state_to_features was not found in self.state_list.\n\
        State attempted to find: {state}"
    )


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
