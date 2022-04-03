from collections import deque
from multiprocessing.sharedctypes import Value
from random import shuffle

import numpy as np

import settings as s


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f"Suitable target found at {best}")
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug("Successfully entered setup code")
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.radius = 3  # how much the radius for the feature vector should be
    self.board_size = self.radius * 2 + 1
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
        "bomb": 10,
    }
    self.idx_to_field = {}


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info("Picking action according to rule set")
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state["field"]
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    coins = game_state["coins"]
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if (
            (arena[d] == 0)
            and (game_state["explosion_map"][d] < 1)
            and (bomb_map[d] > 0)
            and (not d in others)
            and (not d in bomb_xys)
        ):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles:
        valid_actions.append("LEFT")
    if (x + 1, y) in valid_tiles:
        valid_actions.append("RIGHT")
    if (x, y - 1) in valid_tiles:
        valid_actions.append("UP")
    if (x, y + 1) in valid_tiles:
        valid_actions.append("DOWN")
    if (x, y) in valid_tiles:
        valid_actions.append("WAIT")
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history:
        valid_actions.append("BOMB")
    self.logger.debug(f"Valid actions: {valid_actions}")

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ["UP", "DOWN", "LEFT", "RIGHT"]
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [
        (x, y)
        for x in cols
        for y in rows
        if (arena[x, y] == 0)
        and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)
    ]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1):
        action_ideas.append("UP")
    if d == (x, y + 1):
        action_ideas.append("DOWN")
    if d == (x - 1, y):
        action_ideas.append("LEFT")
    if d == (x + 1, y):
        action_ideas.append("RIGHT")
    if d is None:
        self.logger.debug("All targets gone, nothing to do anymore")
        action_ideas.append("WAIT")

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append("BOMB")
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append("BOMB")
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and (
        [arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0
    ):
        action_ideas.append("BOMB")

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if yb > y:
                action_ideas.append("UP")
            if yb < y:
                action_ideas.append("DOWN")
            # If possible, turn a corner
            action_ideas.append("LEFT")
            action_ideas.append("RIGHT")
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if xb > x:
                action_ideas.append("LEFT")
            if xb < x:
                action_ideas.append("RIGHT")
            # If possible, turn a corner
            action_ideas.append("UP")
            action_ideas.append("DOWN")
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == "BOMB":
                self.bomb_history.append((x, y))

            return a


def state_to_features(self, game_state) -> np.array:
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
        10: bomb (without agent standing on it) unless we are standing on our own bomb
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
