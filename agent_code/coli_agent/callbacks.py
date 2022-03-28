import glob
import itertools
import os
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np
from sympy import exp, solve, symbols

from settings import COLS, ROWS

Coordinate = Tuple[int]

Graph = nx.Graph
Action = str

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
SHORTEST_PATH_ACTIONS = ACTIONS[:4]


def setup(self) -> None:
    """Sets up everything. (First call)"""

    self.new_state = None
    self.lattice_graph = nx.grid_2d_graph(m=COLS, n=ROWS)
    self.state_list = list_possible_states()

    # for plotting
    self.q_table_fraction_unseen = []
    self.q_table_average_seen = []
    self.q_table_distribution_of_actions = []

    self.exploration_rates_of_episodes = []
    self.rewards_of_episodes = []
    self.game_scores_of_episodes = []

    # find latest q_table
    list_of_q_tables = glob.glob("./q_tables/*.npy")
    self.latest_q_table_path = max(list_of_q_tables, key=os.path.getctime)
    self.latest_q_table = np.load(self.latest_q_table_path)

    self.logger.info(f"Using q-table: {self.latest_q_table_path}")

    # train if flag is present or if there is no q_table present
    if self.train or not os.path.isfile(self.latest_q_table_path):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = len(self.state_list)

        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.1  # at end of all episodes

        self.exploration_decay_rate = _determine_exploration_decay_rate(self)

        if self.continue_training:
            self.logger.info("Continuing training on latest q_table")
            self.q_table = self.latest_q_table

        else:
            self.logger.info("Starting training from scratch")
            self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))

    else:
        self.logger.info("Using latest Q-Table for testing")
        self.q_table = self.latest_q_table


def list_possible_states() -> List[dict]:
    """Creates a list of dicts of all possible state feature combinations (aka states)

    Makes use of itertools.product(), which produces all possible combinations from lists of options.
    Example: (1, 2), (3, 4, 5) -> (1,3), (1,4), (1,5), (2,3), (2,4), (2, 5)
    The beginning tuples in this case are the possible values for each of our state features.
    The resulting list of tuples gets converted into a list of dicts for easier human access later on.
    """
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
            (False, True),  # in enemy zone
            (
                False,
                True,
            ),  # is it safe to drop a bomb? - if in enemy zone and safe to bomb ... if many crates and safe to bomb ...
            (
                "ZERO",
                "LOW",
                "HIGH",
            ),  # how many surrounding crates are there (and how close are they?) - high bomb dropping reward when this is high
            ("FREE", "BLOCKED"),  # is the next tile in the four directions free to move to
            ("FREE", "BLOCKED"),
            ("FREE", "BLOCKED"),
            ("FREE", "BLOCKED"),
        )
    )

    # Conversion into dict via position in the tuples (becuause of this, order matters)
    state_dicts = []
    for vector in states:
        state_dict = {
            "coin_direction": vector[0],
            "bomb_safety_direction": vector[1],
            "in_enemy_zone": vector[2],
            "safe_to_bomb": vector[3],
            "crate_priority": vector[4],
            "down": vector[5],
            "up": vector[6],
            "right": vector[7],
            "left": vector[8],
        }
        state_dicts.append(state_dict)
    return state_dicts


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    if self.new_state is None:  # is always None in test case
        self.old_state = state_to_features(self, game_state)
    else:  # in train case this is set in game_events_occured
        self.old_state = self.new_state
    state = self.old_state

    # only for logging because here is the easiest place to access this info
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
    self.logger.debug(f"State: {state}")

    # 100% exploitation once we have learnt the q-table/exploit in training
    action = ACTIONS[np.argmax(self.q_table[state])]

    # Alternative: sample from the learnt q_table distribution.
    # if not np.any(self.q_table[state]):
    #     self.logger.debug("Q-Table has all zeros --> choosing random action")
    #     action = np.random.choice(ACTIONS)
    # else:
    #     self.logger.debug("Sampling action from Q-Table")
    #     lowest_q_value_of_state = np.min(self.q_table[state])
    #     non_negative_q_table = self.q_table[state] + abs(lowest_q_value_of_state)
    #     probabilities = [(q_value / sum(non_negative_q_table)) for q_value in non_negative_q_table]
    #     action = np.random.choice(ACTIONS, p=probabilities)

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


def _get_surrounding_tiles(own_coord: Coordinate, n: int) -> List[Coordinate]:
    """Calculates all tiles surrounding self with Manhatten Distance n

    For a starting position, finds every tile that can be reached in n steps,
    i.e. that has a Manhatten Distance of n. Treats all tiles as available tiles,
    even walls, bombs and (some) tiles outside the scope of the board.

    Return only contains unique coordinates and includes starting coordinate.
    """
    own_coord_x = own_coord[0]
    own_coord_y = own_coord[1]
    neighboring_coordinates = []
    for x in range(0, n + 1):
        if x > 15:
            break
        for y in range(0, n + 1 - x):
            if y > 15:
                break
            neighboring_coordinates.append((own_coord_x + x, own_coord_y + y))
            neighboring_coordinates.append((own_coord_x + x, own_coord_y - y))
            neighboring_coordinates.append((own_coord_x - x, own_coord_y + y))
            neighboring_coordinates.append((own_coord_x - x, own_coord_y - y))
    return list(set(neighboring_coordinates))


def _get_neighboring_tiles(own_coord: Coordinate, n: int) -> List[Coordinate]:
    """Calculates n tiles around self in straight line

    For a starting position, finds the n tiles in directions up, down, left right
    (resulting in a total number of 4xn tiles returned).
    Treats all tiles as available tiles, even walls, bombs and tiles outside the scope of the board.

    Returns unique coordinates which don't include starting coordinate.
    """
    own_coord_x = own_coord[0]
    own_coord_y = own_coord[1]
    neighboring_coordinates = []
    for i in range(1, n + 1):
        neighboring_coordinates.append((own_coord_x, own_coord_y + i))  # down in the matrix
        neighboring_coordinates.append((own_coord_x, own_coord_y - i))  # up in the matrix
        neighboring_coordinates.append((own_coord_x + i, own_coord_y))  # right in the matrix
        neighboring_coordinates.append((own_coord_x - i, own_coord_y))  # left in the matrix
    return neighboring_coordinates


def _get_neighboring_tiles_until_wall(
    own_coord: Coordinate, n: int, game_state: dict
) -> List[Coordinate]:
    """Calculates n tiles around self in straight line, stopping at walls

    For a starting position, finds the n tiles in directions up, down, left right.
    Includes all tile types except walls. When the first wall in a direction is detected,
    the following tiles behind that wall aren't included anymore. This equals a
    "bomb radius", e.g. the area in which a bomb can do damage.

    Returns unique coordinates which don't include starting coordinate.
    """
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
                break

        all_good_fields += good_fields

    return all_good_fields


def _get_graph(self, game_state: dict, crates_as_obstacles=True) -> Graph:
    """Converts game_state into a Graph object.

    Every coordinate of a free tile is regarded as a node.

    Considers walls, bombs and active explosions as "non-free" tiles meaning they
    are not regarded as nodes and hence are not part of the graph.

    If crates_as_obstacles is True, crates are regarded as obstacles, else they
    are regarded as free tiles (i.e. they are part of the graph).

    Returns a Graph object.
    """

    if crates_as_obstacles:
        # walls and crates are obstacles
        obstacles = [index for index, field in np.ndenumerate(game_state["field"]) if field != 0]

    else:
        # only walls are obstacles
        obstacles = [index for index, field in np.ndenumerate(game_state["field"]) if field == -1]

    # explosions are always obstacles
    active_explosions = [
        index for index, field in np.ndenumerate(game_state["explosion_map"]) if field != 0
    ]

    # bombs are always obstacles
    bombs = [
        coordinate
        for coordinate, _ in game_state["bombs"]
        if coordinate != game_state["self"][-1]
        and coordinate not in [other_agent[-1] for other_agent in game_state["others"]]
    ]

    obstacles += active_explosions
    obstacles += bombs

    self.logger.debug(f"Obstacles: {obstacles}")

    graph = nx.grid_2d_graph(m=COLS, n=ROWS)

    # inplace operation
    graph.remove_nodes_from(obstacles)  # removes nodes and any edges of all removed nodes
    return graph


def _find_shortest_path(graph: Graph, a: Coordinate, b: Coordinate) -> Tuple[Graph, int]:
    """Calclulates length of shortest path between points a and b in the graph at current time step.

    The calculation is based on the Dijkstra algorithm and is time-independent (i.e. the calculation
    only considers the current state and does not make assumptions about future movements)

    Returns a tuple of the shortest path (as a Graph object that contains the nodes of the shortest
    path) and the length of the shortest path.
    """
    # use Djikstra to find shortest path
    shortest_path = nx.shortest_path(graph, source=a, target=b, weight=None, method="dijkstra")
    shortest_path_length = len(shortest_path) - 1  # because path considers self as part of the path
    return shortest_path, shortest_path_length


def _get_action(self, self_coord: Coordinate, shortest_path: Graph) -> Action:
    """Determines next agent action necessary to follow a given shortest path.

    Given a coordinate and the shortest path (as a Graph object that contains the nodes of the
    shortest path), returns UP, DOWN, LEFT or RIGHT depending on what the next move should be
    in order to follow the path.
    """
    goal_coord = shortest_path[1]  # 0th element is self_coord

    self.logger.info(f"Determined goal at {goal_coord} from shortest path feature")

    # if x-coord is the same
    if self_coord[0] == goal_coord[0]:
        if self_coord[1] + 1 == goal_coord[1]:
            return "DOWN"

        elif self_coord[1] - 1 == goal_coord[1]:
            return "UP"

    # if y-coord is the same
    elif self_coord[1] == goal_coord[1]:
        if self_coord[0] + 1 == goal_coord[0]:
            return "RIGHT"

        elif self_coord[0] - 1 == goal_coord[0]:
            return "LEFT"


def _shortest_path_feature(self, game_state: dict) -> Action:
    """Combines path finding functions to determine direction to nearest coin/crate

    Terminology:
    - reachable: there is a path between a and b and crates are considered as obstacles
    - unreachable: there is a path between a and b but crates are considered as free tiles, i.e. b is not
    immediately reachable b/c there is at least one crate in the way. Note that when computing paths to crates,
    by construction, we consider crates as free tiles. Hence all paths to crates are "unreachable".
    - completely unreachable: there is no path between a and b even if considering crates as free tiles. This
    can happen e.g. if there is an explosion that encircles the agent.

    Computes the direction along the shortest path (can be shortest path to a coin or a crate) as follows:

    - If no collectible coins and no crates exist --> random direction

    - If no collectible coins but a crate exists and it is not completely unreachable --> direction towards nearest crate

    - If one or more collectible coins:

        if all coins are completely unreachable for us:
            --> random direction

        elif all coins are unreachable from our position:
            --> towards unreachable nearest coin (thus towards first crate that's in the way)

        elif exactly one coin is reachable from our position:
            # even though there might be a coin that's much closer but
            # blocked or someone else is closer
            --> towards that coin

        elif more than one coin reachable from our position:
            try:
                --> towards nearest coin that no other agent is closer to

            except there is no coin that our agent is nearest to:
                --> towards nearest coin no matter if it's reachable or not
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

    # no coins on board and no crates
    if not any(safe_coins) and not any(
        [index for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    ):
        return np.random.choice(SHORTEST_PATH_ACTIONS)

    elif not any(safe_coins):
        best = (None, np.inf)

        crates_coordinates = [
            index for index, field in np.ndenumerate(game_state["field"]) if field == 1
        ]

        for crate_coord in crates_coordinates:
            try:
                current_path, current_path_length = _find_shortest_path(
                    graph_with_crates, self_coord, crate_coord
                )

            # in some cases it can happen that a crate is completely unreachable e.g. b/c of explosion
            # even though we don't consider crates themselves as obstacles
            except nx.exception.NetworkXNoPath:
                self.logger.debug("Edge case (completely unreachable crate) occured")
                continue

            # better than distance 1 is impossible - save computation time
            if current_path_length == 1:
                self.logger.info("Standing directly next to crate!")
                return _get_action(self, self_coord, current_path)

            elif current_path_length < best[1]:
                best = (current_path, current_path_length)

        if best == (None, np.inf):
            self.logger.info(
                "There are no coins and all crates are completely unreachable. Choosing random direction."
            )
            return np.random.choice(SHORTEST_PATH_ACTIONS)

        return _get_action(self, self_coord, best[0])

    # there is a safe coin
    else:
        self.logger.info("There is a safe coin (and it is not in an explosion)")
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
                # try finding path to coin from our position
                try:
                    current_path, current_path_length = _find_shortest_path(
                        graph_with_crates, self_coord, coin_coord
                    )
                    current_reachable = False
                except nx.exception.NetworkXNoPath:
                    self.logger.info(
                        "Edge case (completely unreachable coin for us even though crates not \
                        considered as obstacles) occured"
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

            # find shortest path to coin from other agents and consider the closest one
            for other_agent in game_state["others"]:
                best_other_agent = (None, np.inf)
                other_agent_coord = other_agent[3]
                try:
                    (
                        current_path_other_agent,
                        current_path_length_other_agent,
                    ) = _find_shortest_path(graph, other_agent_coord, coin_coord)
                    current_other_agent_reachable = True

                # other_agent can't reach coin
                except nx.exception.NetworkXNoPath:

                    # try finding path to coin of other_agent when not considering crates as obstacles in graph
                    try:
                        (
                            current_path_other_agent,
                            current_path_length_other_agent,
                        ) = _find_shortest_path(graph_with_crates, other_agent_coord, coin_coord)
                        current_other_agent_reachable = False

                    except nx.exception.NetworkXNoPath:
                        self.logger.info(
                            f"Edge case (completely unreachable coin for other agent {other_agent} even \
                            though crates not considered as obstacles) occured"
                        )
                        continue

                # penalize with heuristic of *7* more fields if unreachable
                # 7 since that is approximately the time it takes to drop bomb, take cover and come back
                # assuming there is exactly one crate in between the agent and the coin.
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

        # this happens if all coins are completely unreachable for us
        if not any(shortest_paths_to_coins):
            return np.random.choice(SHORTEST_PATH_ACTIONS)

        # sort our paths [0] ascending by length [1]
        shortest_paths_to_coins.sort(key=lambda x: x[0][1])

        shortest_paths_to_coins_reachable = [
            shortest_path_to_coin[0][2] for shortest_path_to_coin in shortest_paths_to_coins
        ]

        # if none of our [0] shortest paths are actually reachable [2] we just go towards
        # the nearest unreachable one (i.e. to its nearest crate)
        if not any(shortest_paths_to_coins_reachable):
            self.logger.debug("No coin reachable ==> Going towards nearest unreachable one")
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

        # if more than one shortest path is reachable we go towards the one that we are closest
        # to that is also reachable and to which no one else is closer
        for shortest_path_to_coin in shortest_paths_to_coins:

            # we are able to reach it and we are closer
            if (
                shortest_path_to_coin[0][2] is True
                and shortest_path_to_coin[0][1] <= shortest_path_to_coin[1][1]
                and shortest_path_to_coin[0][1]
                != 0  # we are standing on a coin because we spawned on it --> correct would be to "WAIT"
                # but we want to stick to "UP", "DOWN", "LEFT" and "RIGHT" hence we just return second
                # closest coin, assuming the agent will go back in the next time step (only happens in coin-heaven scenario)
            ):
                self.logger.debug(
                    f"We are able to reach coin at {shortest_path_to_coin[0]} and we are closest to it"
                )
                return _get_action(self, self_coord, shortest_path_to_coin[0][0])

        self.logger.info("Fallback Action")
        # unless we are not closest to any of our reachable coins, then we return action that leads us to
        # the coin we are nearest to anyway (no matter if reachable or not)
        # try:
        return _get_action(self, self_coord, shortest_paths_to_coins[0][0][0])

        # it is theoretically possible that coins are not reachable by our agent even if we don't consider
        # crates as obstacles where shortest_paths_to_coins will be empty
        # except IndexError:
        #    return np.random.choice(SHORTEST_PATH_ACTIONS)


def enemy_zone_feature(game_state: dict) -> bool:
    """Determines whether agent is within the 'potential bomb area' of another agent.

    This calculation takes into accounts that walls stop explosions. Assumes that on the
    tile of the other agent there is a freshly dropped bomb and asseses whether it would reach our agent.
    """
    own_position = game_state["self"][-1]
    all_enemy_fields = []
    if_dangerous = []
    for enemy in game_state["others"]:
        neighbours_until_wall = _get_neighboring_tiles_until_wall(
            enemy[-1], 3, game_state=game_state
        )
        if neighbours_until_wall:
            all_enemy_fields += neighbours_until_wall

    if len(all_enemy_fields) > 0:
        for bad_field in all_enemy_fields:
            in_danger = own_position == bad_field
            if_dangerous.append(in_danger)

        return any(if_dangerous)
    else:
        return False


def crate_priority_feature(game_state: dict) -> str:
    """Indicates how many crates there are around our agent.

    Bins the number of crates into three categories to minimize the state space:
    ZERO = None
    FEW = 1-3
    MANY = 4-9 (maximum possible)

    Number of crates is weighted: if a crate is directly next to us,
    it is worth 2.5 crates.
    """
    own_position = game_state["self"][-1]
    neighbours = _get_neighboring_tiles_until_wall(own_position, 3, game_state=game_state)

    crate_counter = 0
    for coord in neighbours:
        if game_state["field"][coord[0]][coord[1]] == 1:
            crate_counter += 1
            # crates directly next to own position are weighted
            if own_position[0] == coord[0] + 1 or own_position[0] == coord[0] - 1:
                crate_counter += 2.5
            elif own_position[1] == coord[1] + 1 or own_position[1] == coord[1] - 1:
                crate_counter += 2.5

    if crate_counter == 0:
        return "ZERO"
    elif 1 <= crate_counter < 4:
        return "LOW"
    elif crate_counter >= 4:
        return "HIGH"


def bomb_safety_direction_feature(self, game_state: dict) -> Action:
    """Indicates direction in which to flee from bomb.

    If we aren't in the future explosion area of a bomb, returns CLEAR.
    If we are in the future explosion area of a bomb, but can't escape anymore,
    returns NO_WAY_OUT.
    Otherwise returns the direction of the next step we should take in order to
    get onto a safe tile.

    """
    own_position = game_state["self"][-1]
    bomb_positions = [bomb[0] for bomb in game_state["bombs"]]

    # radius around agent in which a bomb would hit it
    relevant_neighbors = _get_neighboring_tiles_until_wall(own_position, 3, game_state)
    relevant_neighbors.append(own_position)

    if not any(
        [neighbor in bomb_positions for neighbor in relevant_neighbors]
    ):  # agent is not in any future explosion zone of bomb
        return "CLEAR"

    bomb_explosion_tiles = {
        own_position
    }  # on which tiles will there be an explosion? (always includes self, otherwise would be CLEAR)
    reach = 5  # how far we can still go before most urgent bomb blows up
    for bomb in game_state["bombs"]:
        bomb_explosion_tiles.update(_get_neighboring_tiles_until_wall(bomb[0], 3, game_state))
        if bomb[1] + 2 < reach:
            reach = bomb[1] + 2

    graph = _get_graph(self, game_state)
    available_neighbors = _get_surrounding_tiles(own_position, reach)

    shortest_path = None
    shortest_distance = 1000  # arbitrary high number
    for n in available_neighbors:
        if n not in graph:
            continue
        if n in bomb_explosion_tiles:
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

    if not shortest_path:
        return (
            "NO_WAY_OUT"  # depending on purpose for which func was called, gets converted later on
        )

    self.logger.debug(f"There is a bomb safety goal and the path to it is {shortest_path}")

    return _get_action(self, own_position, shortest_path)


def safe_to_bomb_feature(self, original_game_state: dict) -> bool:
    """Indicates whether dropping a bomb in current position would or would not lead to certain death.

    This is done by creating a fake game state in which there is a bomb in the current agent position
    and then calling the bomb_safety_feature(), which returns NO_WAY_OUT if that situation would mean
    that agent couldn't reach a safe tile in time.
    This is the reason why there is a "NO_WAY_OUT" return of bomb_safety_feature().

    Also considers 'impossible to drop bomb' (i.e. already has an active bomb) as 'unsafe'.

    Return False ==> not safe to drop bomb, return True ==> (probably) safe to drop bomb
    """
    # first position is never safe anyway -> safe computation time
    if original_game_state["step"] == 1:
        return False

    # can't place bomb
    if not original_game_state["self"][2]:
        return False

    # if there was a bomb in current self position, would there be an escape route?
    altered_game_state = deepcopy(original_game_state)
    altered_game_state["bombs"].append((original_game_state["self"][-1], 3))
    if bomb_safety_direction_feature(self, altered_game_state) == "NO_WAY_OUT":
        return False

    return True


def blockage_feature(self, game_state: dict) -> List[str]:
    """Determines whether neighboring tile is free or blocked.

    For each of the four directions, returns FREE or BLOCKED
    depending on what's (not) on the neighboring tile.
    BLOCKED = wall, crate, other agent, bomb, explosion
    FREE = otherwise
    """
    results = ["FREE", "FREE", "FREE", "FREE"]
    own_position = game_state["self"][-1]
    enemy_positions = [enemy[-1] for enemy in game_state["others"]]
    bomb_positions = []
    imminent_explosions = set()

    # accumulates currently active bomb
    for bomb in game_state["bombs"]:
        bomb_positions.append(bomb[0])
        if bomb[1] == 0:
            future_explosion = _get_neighboring_tiles_until_wall(bomb[0], 3, game_state)
            future_explosion += bomb[0]
            imminent_explosions.update(future_explosion)

    # iterates over the four directions
    for i, neighboring_coord in enumerate(_get_neighboring_tiles(own_position, 1)):
        explosion_present = False
        bomb_present = False
        neighboring_x, neighboring_y = neighboring_coord
        neighboring_content = game_state["field"][neighboring_x][
            neighboring_y
        ]  # content of tile, e.g. crate=1
        if neighboring_coord in bomb_positions:
            bomb_present = True
        if neighboring_coord in imminent_explosions:
            explosion_present = True
        if game_state["explosion_map"][neighboring_x][neighboring_y] != 0:
            explosion_present = True
        if (
            neighboring_content != 0
            or neighboring_coord in enemy_positions
            or explosion_present
            or bomb_present
        ):
            results[i] = "BLOCKED"

    return results


def state_to_features(self, game_state: dict) -> int:
    """Main function which converts a game state to a state index.

    Takes a game state dict. Calls all feature functions and plucks results
    into a state dict. Iterates over list of all possible state dicts
    and returns the index of the matching state dict, which is the state
    correponsing to the passed state dict.
    """

    state_dict = {}

    # Feature 1: Next direction in shortest path to nearest coin or crate
    direction = _shortest_path_feature(self, game_state)
    if direction in ["DOWN", "UP", "RIGHT", "LEFT"]:
        state_dict["coin_direction"] = direction
    else:
        raise ValueError(f"Invalid directon to nearest coin/crate: {direction}")

    # Feature 2: Next direction in path to bomb safety zone, or "CLEAR" if not in bomb zone
    bomb_safety_result = bomb_safety_direction_feature(self, game_state)
    if bomb_safety_result in ["DOWN", "UP", "RIGHT", "LEFT", "CLEAR"]:
        state_dict["bomb_safety_direction"] = bomb_safety_result
    elif bomb_safety_result == "NO_WAY_OUT":
        state_dict["bomb_safety_direction"] = np.random.choice(SHORTEST_PATH_ACTIONS)
    else:
        raise ValueError(f"Invalid directon to bomb safety: {bomb_safety_result}")

    # Feature 3: Is agent within bombing reach of other agent?
    state_dict["in_enemy_zone"] = enemy_zone_feature(game_state=game_state)

    # Feature 4: Is it safe to plant a bomb?
    state_dict["safe_to_bomb"] = safe_to_bomb_feature(self, game_state)

    # Feature 5: amount of crates within destruction reach: ZERO, FEW or MANY
    state_dict["crate_priority"] = crate_priority_feature(game_state=game_state)

    # Feature 6-9: blocked down/up/right/left?
    (
        state_dict["down"],
        state_dict["up"],
        state_dict["right"],
        state_dict["left"],
    ) = blockage_feature(self, game_state)

    self.logger.info(f"Feature dict: {state_dict}")

    for i, state in enumerate(self.state_list):
        if state == state_dict:
            return i

    self.logger.debug(f"State list: {self.state_list}")
    raise ReferenceError(
        "State dict created by state_to_features was not found in self.state_list."
    )
