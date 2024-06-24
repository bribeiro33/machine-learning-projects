import math
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from queue import Queue, LifoQueue, PriorityQueue
from enum import Enum
from matplotlib.collections import LineCollection
import os

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Node:
    """The Node class imitates the Node mentioned in lecture.
    A node represents a step along the way of searching for the goal state in
    a search tree/graph.
    """

    def __init__(self, coords, parent=None, action=None, path_cost=0):
        self.coords = coords
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def trace_back(self):
        """Start tracing from this node back to a root node according to parent
        relationship.

        :return: A list of coordinates delineates the path from the starting
        point to the end point, a list of actions taken in the path
        """
        coord_path = []
        actions = []
        trace: Node = self
        while trace is not None:
            coord_path.append(trace.coords)
            if trace.action is not None:
                actions.append(trace.action)
            trace = trace.parent
        coord_path.reverse()
        actions.reverse()
        return coord_path, actions

    def expand(self, grid):
        """For each neighbouring node that can be reached from the current node
        within one action, 'yield' the node.

        Hints:
        Use grid.valid_ordered_action(...) to obtain all valid action for a
        given coordinate in the grid.

        You may use yield to temporarily return a node to the caller
        function (general search function) while saving the context of this
        function. When expand is called again, execution will be resumed from
        where it was left of. Follow this link to learn more:

        https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do


        In Python, to call the polymorphic constructor, you may use
         type(self)([args])

        :param grid: a grid object
        You may use 'grid' to access the action costs
        :return: yield the children Node base (either Node or AStarNode)
        """
        grid.expansion_history.append(self.coords)

        # Your implementation :)
        current_coord = self.coords

        # get all avaliable actions (list)
        actions = grid.valid_ordered_action(self.coords)
        # loop while there are still avaliable actions from that coord
        for act in actions:
            # take that action, where end up
            next_coord = grid.resulting_coord(self.coords, act)
            # cost of that action: node.PATH_COST + problem.action_cost
            total_cost = self.path_cost + grid.action_cost(act) # grid.actions_cost(act) might b wrong
            yield type(self)(next_coord, self, act, total_cost)


    def __eq__(self, other):
        """Used for ordering nodes in the priority queue frontier in UCS."""
        return self.path_cost == other.path_cost

    def __lt__(self, other):
        """Used for ordering nodes in a Priority Queue."""
        return self.path_cost < other.path_cost


class AStarNode(Node):

    # END_COORD is set first before using the heuristics function
    END_COORDS = None

    def __init__(self, coords, parent=None, action=None, cost=0):
        super().__init__(coords, parent, action, cost)
        # store the heuristics value for this node
        self.h_val = self.heuristic_function()

    def heuristic_function(self):
        """Return the straight line distance between the node's coordinate
        and the coordinate of the end point noted in AStarNode.END_COORD
        Your Implementation :)
        """
        straight_line_distance = math.sqrt((self.coords[0] - self.END_COORDS[0])*(self.coords[0] - self.END_COORDS[0]) + (self.coords[1] - self.END_COORDS[1])*(self.coords[1] - self.END_COORDS[1]))
        return straight_line_distance

    def sum_path_cost_heuristics(self):
        """Return the sum of the current path cost and the heuristics value.

        Used in comparing node in the priority queue frontier.
        """
        return self.path_cost + self.h_val

    def __eq__(self, other):
        """Used for comparing nodes in the priority queue frontier for
        AStarNodes."""
        return self.sum_path_cost_heuristics() ==\
               other.sum_path_cost_heuristics()

    def __lt__(self, other):
        """Used in comparing AStarNodes in the frontier."""
        return self.sum_path_cost_heuristics() < other.sum_path_cost_heuristics()

    def __gt__(self, other):
        """Used in comparing AStarNodes in the frontier."""
        return self.sum_path_cost_heuristics() > other.sum_path_cost_heuristics()

class Grid:
    """Storing information about the grid being read."""

    def image_coordinate_to_cartesian(self, row, col):
        """Convert an image coordinate to a cartesian coordinate."""
        return col, self.height - 1 - row

    def cartesian_coordinate_to_memory(self, x, y):
        """Convert an image coordinate to a cartesian coordinate."""
        return self.height - 1 - y, x

    def action_cost(self, action: Action):
        """Given an action. Output the cost of an action."""
        return self.action_costs[action.value]

    def __init__(self, f_name):
        """Parse the text grid input."""
        self.start = None  # record the position of the starting location
        self.end = None  # record the position of the target location
        self.dead_zone = set()  # set of co-ordinates that are "lava"
        self.height = None
        self.width = None
        # cwd = os.getcwd()
        # print(cwd)

        self.action_costs = []
        # a list of nodes that have been expanded
        self.expansion_history = []

        with open(f_name, 'r') as file:
            self.width, self.height = [int(x) for x in file.readline().split()]

            self.action_costs = [int(x) for x in file.readline().split()]
            if len(self.action_costs) != 4:
                raise ValueError("Need to specify the cost of all four actions.")

            row = 0
            for line in file:
                line = re.sub(r'\n', '', line)
                for col, c in enumerate(line):
                    if c == 'A':
                        if self.start is not None:
                            raise ValueError("Cannot have two Start positions")
                        self.start = self.image_coordinate_to_cartesian(row,
                                                                        col)
                    elif c == 'B':
                        if self.end is not None:
                            raise ValueError("Cannot have two End positions")
                        self.end = self.image_coordinate_to_cartesian(row, col)
                    elif c == '*':
                        self.dead_zone.add(
                            self.image_coordinate_to_cartesian(row, col))
                    elif c != '.':
                        print(c)
                        raise ValueError("Unknown character")
                row += 1

        if self.start is None:
            raise ValueError("No starting point")

        if self.end is None:
            raise ValueError("No ending point")

        if self.width <= 0:
            raise ValueError("Invalid width")

        if self.height <= 0:
            raise ValueError("Invalid height")

    def visualize_expansion(self, path):
        """Visualize the grid, expansion history, and a path in matplotlib."""
        fig = plt.figure(figsize=(self.width, self.height))
        plt.subplot(1, 1, 1)
        blocks = np.zeros((self.width, self.height))
        blocks[:] = np.nan
        for co_ord in self.dead_zone:
            blocks[co_ord[1], co_ord[0]] = 2

        expansion_cval = np.zeros((self.width, self.height))

        for i, co_ord in enumerate(self.expansion_history):
            expansion_cval[co_ord[1], co_ord[0]] = \
                len(self.expansion_history) - i + len(self.expansion_history)

        plt.pcolormesh(
            expansion_cval,
                   shading='flat',
                   edgecolors='k', linewidths=1, cmap='Blues')

        cmap = matplotlib.colors.ListedColormap(['grey', 'grey'])

        plt.pcolormesh(
            blocks,
            shading='flat',
            edgecolors='k', linewidths=1, cmap=cmap)

        segments = [
            (
                (path[i][0] + 0.5, path[i][1] + 0.5),
                (path[i+1][0] + 0.5, path[i+1][1] + 0.5)
            )
            for i in range(len(path) - 1)
        ]

        lc = LineCollection(segments, colors='r', linewidths=10)
        plt.gca().add_collection(lc)
        plt.show()

    def goal_test(self, node: Node):
        """Determines whether the 'goal' of this search has been reached
        depending on the given node.
        
        Your Implementation :)
        """
        return node.coords == self.end


    def clear_expansion_history(self):
        self.expansion_history.clear()

    @staticmethod
    def resulting_coord(starting_coord, action: Action):
        """Return the resulting coordinate given a starting coordinate and
        an action."""
        if action == Action.LEFT:
            return starting_coord[0] - 1, starting_coord[1]

        if action == Action.RIGHT:
            return starting_coord[0] + 1, starting_coord[1]

        if action == Action.UP:
            return starting_coord[0], starting_coord[1] + 1

        if action == Action.DOWN:
            return starting_coord[0], starting_coord[1] - 1

    def valid_coord(self, coord):
        """Return a boolean indicating whether the coordinate is
        valid coordinate within the grid.
        i.e. within the boundary of the grid and not a *.
        """
        x, y = coord
        return (x >= 0) and (x < self.width) and (y >= 0) and (y < self.height) \
               and coord not in self.dead_zone

    def valid_ordered_action(self, coord):
        """For a specific coordinate, return a list of valid actions
        the robot from a given coordinate within the grid.
        For all valid actions, they are ordered by "LEFT", "RIGHT", "UP", "DOWN".

        :return: a list of valid actions ordered by their enum values
        """
        valid_actions = []
        for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
            result_coord = Grid.resulting_coord(coord, action)
            if self.valid_coord(result_coord):
                valid_actions.append(action)

        return valid_actions

    def bfs(self):
        """Implement breadth-first-search here:

        :return: a tuple of three items:
        a boolean indicating whether there exists a path
        a tuple of list of coordinates from the
            starting point to the end point, and the actions taken or None when 
            there exists no path
        the list of coordinates of Nodes that are expanded during the search.
        """
        self.clear_expansion_history()
        # Your Implementation :)
        found_path = True
        initial_node = Node(self.start)
        frontier = Queue(maxsize = self.height * self.width) # change to number of states from reading file info
        frontier.put(initial_node)
        reached = {tuple(initial_node.coords): initial_node}
        if self.goal_test(initial_node):
            path = ([self.start], None) # is None correct here?
            return (found_path, path, self.expansion_history)
        while not frontier.empty():
            current_node = frontier.get()
            for child in current_node.expand(self):
                if self.goal_test(child):
                    path = child.trace_back()
                    return found_path, path, self.expansion_history
                child_coords = tuple(child.coords)
                if (not (child_coords in reached)) or (child.path_cost < reached[child_coords].path_cost):
                    reached[child_coords] = child
                    frontier.put(child) 
        found_path = False
        return (found_path, None, self.expansion_history)





    def search(self, start_node, frontier):
        """Strongly recommend using a general search method.
        This function intends to emulate general search architecture function
        mentioned in lecture notes.

        :param start_node: Either a Node or an AStarNode
        :param frontier: a FIFO queue, LIFO queue or a PriorityQueue
        :return: a tuple of three items:
        a boolean indicating whether there exists a path
        a tuple of list of coordinates from the
            starting point to the end point, and the actions taken or None when 
            there exists no path
        the list of coordinates of Nodes that are expanded during the search.
        """

        self.clear_expansion_history()
        
        # Your Implementation :)
        found_path = True
        frontier.put(start_node)
        reached = {tuple(start_node.coords): start_node}
  
        while not frontier.empty():
            current_node = frontier.get()
            if self.goal_test(current_node):
                    path = current_node.trace_back()
                    return found_path, path, self.expansion_history
            for child in current_node.expand(self):
                child_coords = tuple(child.coords)
                if (not (child_coords in reached)) or (child.path_cost < reached[child_coords].path_cost):
                    reached[child_coords] = child
                    frontier.put(child) 
        found_path = False
        return (found_path, None, self.expansion_history)

    def dfs(self):
        """Implement depth first search.
        :return: a tuple of three items:
        a boolean indicating whether there exists a path
        a tuple of list of coordinates from the
            starting point to the end point, and the actions taken or None when 
            there exists no path
        the list of coordinates of Nodes that are expanded during the search.
        
        Your implementation :)
        """
        start_node = Node(self.start)
        frontier = LifoQueue(maxsize=self.height * self.width)
        return self.search(start_node, frontier)

    def ucs(self):
        """Implement uniform cost search.
        :return: a tuple of three items:
        a boolean indicating whether there exists a path
        a tuple of list of coordinates from the
            starting point to the end point, and the actions taken or None when 
            there exists no path
        the list of coordinates of Nodes that are expanded during the search.
        
        Your implementation :)
        """
        # In Node, we have defined Nodes with lower path cost has a higher ‘priority’ – Python has a built-in Priority Queue data structure in the queue module
        start_node = Node(self.start)
        frontier = PriorityQueue(maxsize=self.height * self.width)
        return self.search(start_node, frontier)

    def astar(self):
        """Implement A-Star Search.
        
        :return: a tuple of three items:
        a boolean indicating whether there exists a path
        a tuple of list of coordinates from the
            starting point to the end point, and the actions taken or None when 
            there exists no path
        the list of coordinates of Nodes that are expanded during the search.
        """
        AStarNode.END_COORDS = self.end

        start_node = AStarNode(self.start)
        frontier = PriorityQueue(maxsize=self.height * self.width)
        return self.search(start_node, frontier)