import glob
import itertools
import os
from collections import deque
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
SHORTEST_PATH_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


def setup(self):
    """Sets up everything. (First call)"""

    self.new_state = None
    self.history = deque(maxlen=5)  # tiles visited
    self.lattice_graph = nx.grid_2d_graph(m=COLS, n=ROWS)
    self.previous_distance = 0
    self.current_distance = 0
    self.state_list = list_possible_states()
    self.coins = set()

    # for plotting
    self.q_table_fraction_unseen = []
    self.q_table_average_seen = []
    self.q_table_distribution_of_actions = []

    self.exploration_rates_of_episodes = []
    self.rewards_of_episodes = []
    self.game_scores_of_episodes = []

    # find latest q_table
    list_of_q_tables = glob.glob("q_tables/*.npy")  # * means all if need specific format then *.csv
    self.latest_q_table_path = max(list_of_q_tables, key=os.path.getctime)
    # self.latest_q_table_path = "q_table-2022-03-14T162802-node45.npy"
    self.latest_q_table = np.load(self.latest_q_table_path)

    self.logger.info(f"Using q-table: {self.latest_q_table_path}")

    # train if flag is present or if there is no q_table present
    if self.train or not os.path.isfile(self.latest_q_table_path):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        # self.number_of_states = 1536  # TODO: make this dynamic
        self.number_of_states = 3840
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
            (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3, 4), (0, 1, 2)
        )
    )

    state_dicts = []
    for vector in states:
        state_dict = {
            "neighbours_down": vector[0],
            "neighbours_up": vector[1],
            "neighbours_right": vector[2],
            "neighbours_left": vector[3],
            "current_field": vector[4],
            "game_mode": vector[5],
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
    # print(state)
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
    self.logger.debug(f"[TV] Coins {self.coins}")
    self.logger.debug(f'[TV] Bombs {game_state["bombs"]}')

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


def hot_field_feature(self, game_state: dict) -> int:
    own_position = game_state["self"][-1]
    all_hot_fields, if_dangerous = [], []

    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_pos = bomb[0]  # coordinates of bomb as type tuple
            neighbours_until_wall = get_neighboring_tiles_until_wall(
                bomb_pos, 3, game_state=game_state
            )
            if neighbours_until_wall:
                all_hot_fields += neighbours_until_wall
                all_hot_fields += [bomb_pos]

        if len(all_hot_fields) > 0:
            for lava in all_hot_fields:
                in_danger = own_position == lava
                # if in_danger:
                #     self.logger.debug(f'own {own_position}, lava {lava}')
                if_dangerous.append(in_danger)

            return int(any(if_dangerous))
    else:
        return 0


def blockage_feature(game_state: dict) -> List[int]:
    own_position = game_state["self"][-1]
    enemy_positions = [enemy[-1] for enemy in game_state["others"]]
    results = [0, 0, 0, 0]

    for i, neighboring_coord in enumerate(_get_neighboring_tiles(own_position, 1)):
        neighboring_x, neighboring_y = neighboring_coord
        neighboring_content = game_state["field"][neighboring_x][
            neighboring_y
        ]  # content of tile, e.g. crate=1
        explosion = (
            True if game_state["explosion_map"][neighboring_x][neighboring_y] != 0 else False
        )
        ripe_bomb = False  # "ripe" = about to explode
        if (neighboring_coord, 0) in game_state["bombs"] or (
            neighboring_coord,
            1,
        ) in game_state["bombs"]:
            ripe_bomb = True
        if (
            neighboring_content != 0
            or neighboring_coord in enemy_positions
            or explosion
            or ripe_bomb
        ):
            results[i] = 1
    return results


def progression_feature(self) -> int:
    # self.logger.debug(f"[TV] History {self.history}")
    num_visited_tiles = len(self.history)  # history contains agent coords of last 5 turns
    if num_visited_tiles > 1:  # otherwise the feature is and is supposed to be 0 anyway
        num_unique_visited_tiles = len(set(self.history))
        # of 5 tiles, 3 should be new -> 60%. for start of the episode: 2 out of 2, 2 out of 3, 3 out of 4
        return 1 if (num_unique_visited_tiles / num_visited_tiles) >= 0.6 else 0
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


def bomb_safety_direction_feature(self, game_state) -> Action:
    """
    Issues with this feature:
    * shortest path is in this case not necessarily best path
    * shortest path might change frequently because enemy is moving away -> leads to moving back and forth
    """
    own_position = game_state["self"][-1]
    relevant_neighbors = get_neighboring_tiles_until_wall(own_position, 3, game_state)
    relevant_neighbors.append(own_position)
    bomb_positions = [bomb[0] for bomb in game_state["bombs"]]
    self.logger.debug(f"Bomb positions: {bomb_positions}")

    if not any(
        [neighbor in bomb_positions for neighbor in relevant_neighbors]
    ):  # agent is not in any future explosion zone of bomb
        return "CLEAR"

    bomb_explosion_tiles = [own_position]
    reach = 4  # how far we can still go before most urgent bomb blows up
    for bomb in game_state["bombs"]:
        bomb_explosion_tiles += get_neighboring_tiles_until_wall(bomb[0], 3, game_state)
        if bomb[1] + 1 < reach:
            reach = bomb[1] + 1
    self.logger.debug(f"Reach: {reach}")
    graph = _get_graph(self, game_state)
    available_neighbors = _get_surrounding_tiles(own_position, reach)

    shortest_path = None
    shortest_distance = 1000
    for n in available_neighbors:
        if n not in graph or n in bomb_explosion_tiles:
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
        return "NO_WAY_OUT"  # gets converted into random action for bomb_safety_direction feature

    self.logger.debug(f"There is a bomb safety goal and it is: {shortest_path[1]}")  # ?

    return _get_action(self, own_position, shortest_path)


def neighbouring_field_feature(self, game_state):
    neighbours_result = [0, 0, 0, 0]
    own_position = game_state["self"][-1]
    bombs_pos = [bomb[0] for bomb in game_state["bombs"]]  # bombs position
    active_explosions = [
        index
        for index, field in np.ndenumerate(game_state["explosion_map"])
        if field != 0  # active explosions
    ]
    opponents = [op[-1] for op in game_state["others"]]

    # Aileens code + Trangs edit
    bomb_explosion_tiles = []  # bomb zone
    for bomb in game_state["bombs"]:
        bomb_explosion_tiles += get_neighboring_tiles_until_wall(bomb[0], 3, game_state)

    opponents_tiles = []  # opponents zone
    for bomb in game_state["others"]:
        opponents_tiles += get_neighboring_tiles_until_wall(bomb[-1], 3, game_state)

    graph = _get_graph(self, game_state)
    graph_with_crates = _get_graph(self, game_state, crates_as_obstacles=False)

    shortest_coin_path = None
    shortest_coin_distance = 1000
    shortest_crate_path = None
    shortest_crate_distance = 1000

    neighbours = _get_neighboring_tiles(own_position, 1)
    for idx, nei in enumerate(neighbours):
        if game_state["field"][nei[0]][nei[1]] == 0:  # if free field
            # free_tiles = 0

            for coin in game_state["coins"]:
                if coin not in graph or coin in bomb_explosion_tiles:
                    continue
                try:
                    (
                        current_shortest_path,
                        current_shortest_distance,
                    ) = _find_shortest_path(  # find shortest path to each coin
                        graph, own_position, coin
                    )
                    if (
                        current_shortest_distance < shortest_coin_distance
                    ):  # find the overall shortest path to a coin
                        shortest_coin_path = current_shortest_path
                        shortest_coin_distance = current_shortest_distance

                except nx.exception.NetworkXNoPath:
                    continue

            if not shortest_coin_path:  # if no coin available
                crates_coordinates = [
                    index for index, field in np.ndenumerate(game_state["field"]) if field == 1
                ]
                for crate in crates_coordinates:
                    if crate not in graph_with_crates or crate in bomb_explosion_tiles:
                        continue
                    try:
                        current_shortest_path, current_shortest_distance = _find_shortest_path(
                            graph_with_crates,
                            own_position,
                            crate,  # find shortest path to each coin
                        )
                        if (
                            current_shortest_distance < shortest_crate_distance
                        ):  # find the overall shortest path to a coin
                            shortest_crate_path = current_shortest_path
                            shortest_crate_distance = current_shortest_distance

                    except nx.exception.NetworkXNoPath:
                        continue
                # if all coins collected, search for opponent
                if shortest_crate_path is not None:
                    if shortest_crate_path[1] == nei:  # if next field in shortest path to a coin
                        neighbours_result[idx] = 1
                        if len(shortest_crate_path[1:]) >= 1:
                            self.logger.debug(
                                f"Next field leading to nearest crate: {shortest_crate_path[1]}"
                            )  # ?
                        elif len(shortest_crate_path[1:]) == 1:
                            self.logger.debug(
                                f"Standing next to crate {shortest_crate_path[1]}"
                            )  # ?

            if shortest_coin_path is not None:
                try:
                    if shortest_coin_path[1] == nei:  # if next field in shortest path to a coin
                        neighbours_result[idx] = 1
                        self.logger.debug(
                            f"Next field leading to nearest coin: {shortest_coin_path[1]}"
                        )  # ?

                except IndexError:
                    self.logger.debug(f"Standing on coin: {shortest_coin_path[0]}")  # ?
                    continue

            if nei in bomb_explosion_tiles or nei in opponents_tiles:  # if in bomb or opponent zone
                neighbours_result[idx] = 3

            # next_neighs = _get_neighboring_tiles(nei, 1)
            # for next_nei in next_neighs:
            #     if game_state['field'][next_nei[0]][next_nei[1]] == 0 and next_nei not in bomb_explosion_tiles \
            #         and next_nei not in opponents_tiles: # if in bomb/opponent zone
            #         free_tiles += 1
            #         continue

            # if free_tiles < 2: # if less than two safe fields in neighbours of next field
            #     neighbours_result[idx] = 3

            if nei in bombs_pos or nei in opponents or nei in active_explosions:
                neighbours_result[idx] = 2

        elif game_state["field"][nei[0]][nei[1]] != 0:
            neighbours_result[idx] = 2

    self.logger.debug(f"[down, up, right, left]: {neighbours_result}")

    return neighbours_result


def _get_safe_tiles(self, source_coord, game_state):
    own_position = source_coord
    opponents_pos = [op[-1] for op in game_state["others"]]
    bombs_pos = [b[0] for b in game_state["bombs"]]

    active_explosions = [
        index for index, field in np.ndenumerate(game_state["explosion_map"]) if field != 0
    ]

    safe_tiles = 0
    safe_tiles_coords = []
    crate = 0
    crate_coords = []
    opponents_killed = 0

    directions = ["N", "E", "S", "W"]
    own_coord_x, own_coord_y = own_position[0], own_position[1]

    for d, _ in enumerate(directions):
        crate_in_the_way = False
        for i in range(1, 3 + 1):
            try:
                if directions[d] == "N":
                    if game_state["field"][own_coord_x][own_coord_y - i] == 1:
                        crate += 1
                        crate_in_the_way = True
                    elif (
                        game_state["field"][own_coord_x][own_coord_y - i] == 0
                        and (own_coord_x, own_coord_y - i) not in bombs_pos
                        and (own_coord_x, own_coord_y - i) not in active_explosions
                        and not crate_in_the_way
                    ):
                        if game_state["field"][own_coord_x - 1][own_coord_y - i] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x - 1, own_coord_y - i))
                        if game_state["field"][own_coord_x + 1][own_coord_y - i] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x + 1, own_coord_y - i))
                        if i == 3:
                            if game_state["field"][own_coord_x][own_coord_y - i - 1] == 0:
                                safe_tiles += 1
                                safe_tiles_coords.append((own_coord_x, own_coord_y - i - 1))
                        # crate_coords.append((own_coord_x, own_coord_y - i))

                    # if (own_coord_x, own_coord_y - i) in opponents_pos:
                    #     opponents_killed += 1

                    # break
                    # crate_in_the_way = True

                    else:
                        break

                if directions[d] == "E":
                    if game_state["field"][own_coord_x + i][own_coord_y] == 1:
                        crate += 1
                        crate_in_the_way = True
                    elif (
                        game_state["field"][own_coord_x + i][own_coord_y] == 0
                        and (own_coord_x + i, own_coord_y) not in bombs_pos
                        and (own_coord_x + i, own_coord_y) not in active_explosions
                        and not crate_in_the_way
                    ):
                        if game_state["field"][own_coord_x + i][own_coord_y - 1] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x + i, own_coord_y - 1))
                        if game_state["field"][own_coord_x + i][own_coord_y + 1] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x + i, own_coord_y + 1))
                        if i == 3:
                            if game_state["field"][own_coord_x + i + 1][own_coord_y] == 0:
                                safe_tiles += 1
                                safe_tiles_coords.append((own_coord_x + i + 1, own_coord_y))

                    # elif (own_coord_x + 1, own_coord_y) in opponents_pos:
                    #     opponents_killed += 1

                    # break
                    # crate_coords.append((own_coord_x, own_coord_y - i))
                    else:
                        break

                if directions[d] == "S":
                    if game_state["field"][own_coord_x][own_coord_y + i] == 1:
                        crate += 1
                        crate_in_the_way = True
                    elif (
                        game_state["field"][own_coord_x][own_coord_y + i] == 0
                        and (own_coord_x, own_coord_y + i) not in bombs_pos
                        and (own_coord_x, own_coord_y + i) not in active_explosions
                        and not crate_in_the_way
                    ):
                        if game_state["field"][own_coord_x - 1][own_coord_y + i] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x - 1, own_coord_y + i))
                        if game_state["field"][own_coord_x + 1][own_coord_y + i] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x + 1, own_coord_y + i))
                        if i == 3:
                            if game_state["field"][own_coord_x][own_coord_y + i + 1] == 0:
                                safe_tiles += 1
                                safe_tiles_coords.append((own_coord_x, own_coord_y + i + 1))

                    # elif (own_coord_x, own_coord_y + i) in opponents_pos:
                    #     opponents_killed += 1
                    else:
                        break

                if directions[d] == "W":
                    if game_state["field"][own_coord_x - i][own_coord_y] == 1:
                        crate += 1
                        crate_in_the_way = True
                    elif (
                        game_state["field"][own_coord_x - i][own_coord_y] == 0
                        and (own_coord_x - i, own_coord_y) not in bombs_pos
                        and (own_coord_x - i, own_coord_y) not in active_explosions
                        and not crate_in_the_way
                    ):
                        if game_state["field"][own_coord_x - i][own_coord_y - 1] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x - i, own_coord_y - 1))
                        if game_state["field"][own_coord_x - i][own_coord_y + 1] == 0:
                            safe_tiles += 1
                            safe_tiles_coords.append((own_coord_x - i, own_coord_y + 1))
                        if i == 3:
                            if game_state["field"][own_coord_x - i - 1][own_coord_y] == 0:
                                safe_tiles += 1
                                safe_tiles_coords.append((own_coord_x - i - 1, own_coord_y))

                    # elif (own_coord_x - 1, own_coord_y) in opponents_pos:
                    #     opponents_killed += 1
                    else:
                        break

            except IndexError:
                # print("Border")
                break

    return crate, safe_tiles_coords, opponents_killed


def current_field_feature(self, game_state):
    own_position = game_state["self"][-1]
    opponents_pos = [op[-1] for op in game_state["others"]]
    bombs_pos = [b[0] for b in game_state["bombs"]]
    safe_tiles_wrt_bombs = []
    safe_tiles_coords = []

    bombs_dict = {}
    for b in game_state["bombs"]:
        neighbours = get_neighboring_tiles_until_wall(b[0], 3, game_state=game_state)
        bombs_dict[b[0]] = [neighbours, b[1]]

    danger = False
    for (pos, content) in bombs_dict.items():
        # self.logger.debug(f"own: {own_position}, neis: {content[0]}, countdown: {content[1]}")
        if (own_position == pos) or (own_position in content[0]):  #  and (content[1] == 0):
            danger = True
            self.logger.debug(f"Haha, I'm in danger.")
            break

    crate, safe_tiles_coords, opponents_killed = _get_safe_tiles(self, own_position, game_state)

    if danger:
        safe_tiles_wrt_bombs = []
        for pos, content in bombs_dict.items():
            # if (own_position in content[0] or own_position == pos) and own_position[1] == pos[1]:  # if same y-axis
            if own_position == pos:
                self.logger.debug(f"own bomb was placed on {pos}")
                _, safe_tiles_wrt_bomb, _ = _get_safe_tiles(self, pos, game_state)
                for tile in safe_tiles_wrt_bomb:
                    safe_tiles_wrt_bombs.append(tile)

            elif own_position in content[0]:
                if own_position[1] == pos[1]:  # if same y-axis
                    if own_position[0] < pos[0]:
                        _, safe_tiles_wrt_bomb, _ = _get_safe_tiles(self, pos, game_state)
                        for tile in safe_tiles_wrt_bomb:
                            if tile[0] < pos[0]:
                                # self.logger.debug(f'tile on the left of bomb {tile}')
                                safe_tiles_wrt_bombs.append(tile)
                    elif own_position[0] > pos[0]:
                        _, safe_tiles_wrt_bomb, _ = _get_safe_tiles(self, pos, game_state)
                        for tile in safe_tiles_wrt_bomb:
                            if tile[0] > pos[0]:
                                # self.logger.debug(f'tile on the right of bomb {tile}')
                                safe_tiles_wrt_bombs.append(tile)
                # elif (own_position in content[0] or own_position == pos) and own_position[0] == pos[0]:  # if same y-axis
                elif own_position[0] == pos[0]:  # if same x-axis
                    if own_position[1] < pos[1]:
                        _, safe_tiles_wrt_bomb, _ = _get_safe_tiles(self, pos, game_state)
                        for tile in safe_tiles_wrt_bomb:
                            if tile[1] < pos[1]:
                                safe_tiles_wrt_bombs.append(tile)
                                # self.logger.debug(f'tile above bomb {tile}')
                    elif own_position[1] > pos[1]:
                        _, safe_tiles_wrt_bomb, _ = _get_safe_tiles(self, pos, game_state)
                        for tile in safe_tiles_wrt_bomb:
                            if tile[1] > pos[1]:
                                safe_tiles_wrt_bombs.append(tile)
                                # self.logger.debug(f'tile under bomb {tile}')

    safe_tiles = len(safe_tiles_coords)

    self.logger.debug(
        f"crates: {crate}, safe tiles with no crates in between: {safe_tiles_coords}, ops: {opponents_killed}"
    )
    self.logger.debug(
        f"crates: {crate}, safe tiles wrt bombs: {safe_tiles_wrt_bombs}, ops: {opponents_killed}"
    )

    if (own_position in bombs_dict.keys()) or danger:
        current_field = 4

    if (crate >= 6 or opponents_killed > 1) and safe_tiles > 0 and game_state["self"][2] == True:
        current_field = 3

    elif (crate >= 3 or opponents_killed > 1) and safe_tiles > 0 and game_state["self"][2] == True:
        current_field = 2

    elif (crate >= 1 or opponents_killed > 1) and safe_tiles > 0 and game_state["self"][2] == True:
        current_field = 1

    if (
        not danger
        and game_state["self"][2] == False
        or game_state["self"][2] == True
        and (safe_tiles == 0 or crate == 0)
    ):
        self.logger.debug(f"Not in danger and no bomb function, or no safe tiles or no crates")
        current_field = 0

    self.logger.debug(f"Current field feature {current_field}")

    return current_field


def mode_feature(self, game_state):
    if len(game_state["coins"]) > 0:
        return 0
    elif len(game_state["coins"]) == 0:
        return 1
    elif len(self.coins) == 9 and len(game_state["coins"]) == 0 and len(game_state["others"]) != 0:
        return 2


def state_to_features(self, game_state) -> np.array:
    """Parses game state to features"""

    state_dict = {
        "neighbours_down": None,
        "neighbours_up": None,
        "neighbours_right": None,
        "neighbours_left": None,
        "current_field": None,
        "game_mode": None,
    }

    # Feature 1: if on hot field or not
    # state_dict["bomb_danger_zone"] = hot_field_feature(self, game_state=game_state)

    # Feature 2-5 ("Blockages")
    # (
    #     state_dict["blocked_down"],
    #     state_dict["blocked_up"],
    #     state_dict["blocked_right"],
    #     state_dict["blocked_left"],
    # ) = blockage_feature(game_state=game_state)

    # New feature 1-4 ('Kind of neighbours')
    (
        state_dict["neighbours_down"],
        state_dict["neighbours_up"],
        state_dict["neighbours_right"],
        state_dict["neighbours_left"],
    ) = neighbouring_field_feature(self, game_state=game_state)

    # New feature 5 ('Current field')
    state_dict["current_field"] = current_field_feature(self, game_state=game_state)

    # New feature 6 ('Game mode')
    state_dict["game_mode"] = mode_feature(self, game_state=game_state)

    # Feature 6 ("Going to new tiles")
    # state_dict["progressed"] = progression_feature(self)

    # Feature 7: Next direction in shortest path to coin or crate
    direction = _shortest_path_feature(self, game_state)
    # same order as features 2-5
    # if direction in ["DOWN", "UP", "RIGHT", "LEFT"]:
    #     state_dict["coin_direction"] = direction
    # else:
    #     self.logger.debug(f"Shortest path feature produced invalid return: {direction}")
    #     raise ValueError("Invalid directon to nearest coin/crate")

    # Feature 8: amount of crates within destruction reach: small: 0, medium: 1<4, high: >= 4
    # state_dict["surrounding_crates"] = surrounding_crates_feature(game_state=game_state)

    # Feature 9: if in opponents area
    # state_dict["enemy_danger_zone"] = enemy_zone_feature(game_state=game_state)

    self.logger.info(f"Feature dict: {state_dict}")

    for i, state in enumerate(self.state_list):
        if state == state_dict:
            return i

    raise ReferenceError("State dict created by state_to_features was not found in self.state_list")


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
