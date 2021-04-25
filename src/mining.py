#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

class problem with

An open-pit mine is a grid represented with a 2D or 3D numpy array.

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

A state indicates for each surface location  how many cells
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.
"""

import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools
import functools  # @lru_cache(maxsize=32)
import time
import search

from numbers import Number


def convert_to_tuple(a):
    """
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.

    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    """
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)


def convert_to_list(a):
    """
    Convert the array-like parameter 'a' into a nested list of the same
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists
    """
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]


def my_team():
    """
    Return the list of the team members of this assignment submission
    as a list of triplet of the form (student_number, first_name, last_name)
    """
    return [(9193243, 'Brodie', 'Smith'),
            (10250191, 'Keith', 'Hall'),
            (10273913, 'Sy', 'Ha')]


class Mine(search.Problem):
    """
    Mine represent an open mine problem defined by a grid of cells
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.

    The z direction is pointing down, the x and y directions are surface
    directions.

    An instance of a Mine is characterized by
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between
                           adjacent columns

    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the
                                         mine

    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.

    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.

    States must be tuple-based.
    """
    def __init__(self, underground, dig_tolerance=1):
        '''
        Constructor

        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y,
        self.len_z, self.cumsum_mine, and self.initial

        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.
        '''
        assert underground.ndim in (2, 3)

        # self.underground  should be considered as a 'read-only' variable!
        self.underground = underground

        self.dig_tolerance = dig_tolerance

        self.is_2D = False if underground.ndim == 3 else True

        # Mine lengths
        self.len_x = underground.shape[0]
        self.len_z = (underground.shape[1] if self.is_2D
                      else underground.shape[2])
        self.len_y = None if self.is_2D else underground.shape[1]

        # The sum of the mine heading down the Z axis (digging down)
        self.cumsum_mine = np.cumsum(underground,
                                     axis=-1,
                                     dtype=float)

        # The initial undug state of the mine.
        self.initial = np.zeros((self.len_x if self.is_2D else
                                (self.len_x, self.len_y)),
                                dtype=int)
        self.initial = convert_to_tuple(self.initial)

        # --------------------- init for BB--------------------
        # Initial best goal state
        self.best = self.initial
        # A large starting highest which will be driven down
        self.highest = 1000000000

    def surface_neighbours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.
        '''
        L = []
        assert len(loc) in (1, 2)
        if len(loc) == 1:
            # Ensure we have a loc within the state array.
            assert loc[0] < self.len_x
            if loc[0] - 1 >= 0:
                L.append((loc[0] - 1,))
            if loc[0] + 1 < self.len_x:
                L.append((loc[0] + 1,))
        else:
            # len(loc) == 2
            assert loc[0] < self.len_x
            assert loc[1] < self.len_y
            for dx, dy in ((-1, -1), (-1, 0), (-1, +1),
                           (0, -1), (0, +1),
                           (+1, -1), (+1, 0), (+1, +1)):
                if ((0 <= loc[0] + dx < self.len_x) and
                   (0 <= loc[1] + dy < self.len_y)):
                    L.append((loc[0] + dx, loc[1] + dy))
        return L

    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't break the dig_tolerance constraint.

        Parameters
        ----------
        state :
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions
        '''
        state = np.array(state)

        def pass_to(state, x, y=None):
            """
            This is a local helper function for the generators.
            :Param state: The np.array of the state.
            :Param x: The current x coord we are testing.
            :Param y: The current y coord we are testing if required.
            """
            updated = state.copy()
            # Check to see if we have dug to far down.
            if self.is_2D:  # Its a 1D state we are testing.
                np.add.at(updated, x, 1)
                if updated[x] > self.len_z:
                    return True
            else:
                np.add.at(updated, (x, y), 1)
                if updated[x, y] > self.len_z:
                    return True
            # Else test if it is dangerous.
            return self.is_dangerous(updated)

        # The generators which will return a valid action.
        if self.is_2D:
            return ((x,) for x, z in zip(np.arange(self.len_x), state)
                    if not pass_to(state, x))
        else:
            return ((x, y) for x, z in zip(np.arange(self.len_x), state[1])
                    for y in np.arange(len(state[1])) if not
                    pass_to(state, x, y))

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state).
        """
        action = tuple(action)
        new_state = np.array(state)  # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)

    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.
        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())

    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                             + str(self.underground[..., z]) for z in
                             range(self.len_z))

    @staticmethod
    def plot_state(state):
        if state.ndim == 1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]),
                   state
                   )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim == 2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x)  # cols, rows
            x, y = _xx.ravel(), _yy.ravel()
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3, 3))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        :Param state: a tuple/list of the state to be tested.

        :Returns {float}: Of the calculated payoff.
        '''
        # convert to np.array in order to use tuple addressing
        # state[loc]   where loc is a tuple

        assert not isinstance(state, search.Node)
        state = np.array(state)

        if self.is_2D:
            # Create a mask using the state and then sum
            mask = state[:, None] > np.arange(self.underground.shape[1])
            return np.sum(self.underground, where=mask)
        else:
            # As above
            mask = state[:, :, None] > np.arange(self.underground.shape[2])
            return np.sum(self.underground, where=mask)

    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.
        :Param state: A tuple or list of the given state to be tested.

        :Returns {bool}: True if dangerous else False.
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)

        # get the diff along the x dim
        x_diff = np.greater(np.abs(np.diff(state, axis=0)),
                            self.dig_tolerance)
        if x_diff.any():
            return True

        if not self.is_2D:
            # get the diff along the y dim
            y_diff = np.greater(np.abs(np.diff(state, axis=1)),
                                self.dig_tolerance)
            if y_diff.any():
                return True
            # now work out the diff for the diagonals
            diag_1 = np.greater(np.abs((state[:-1, :-1] - state[1:, 1:])),
                                self.dig_tolerance)
            n_state = np.rot90(state)
            diag_2 = np.greater(np.abs((n_state[:-1, :-1] - n_state[1:, 1:])),
                                self.dig_tolerance)
            if diag_1.any() or diag_2.any():
                return True
        return False

    def goal_test(self, state):
        """
        A simple goal test to update the best payoff for use in the BB
        function.
        :Param state: A state Tuple/list or array.

        :Returns {float}: The best states cost value.
        """
        if self.payoff(state) > self.payoff(self.best):
            self.best = state
            self.highest = self.b(state)
        return self.highest

    def b(self, node):
        """
        This is the cost function which uses a state and self.cumsum_mine
        to establish a upper for the branches.
        :Param node: either a Node or state tuple.

        :Return {float}: The estimated cost for the node.
        """
        if isinstance(node, search.Node):
            state = np.array(node.state)
        else:
            state = np.array(node)

        # Create a mask removing current state values
        if self.is_2D:
            mask = state[:, None] > np.arange(self.underground.shape[1])
        else:
            mask = state[:, :, None] > np.arange(self.underground.shape[2])
        # Apply the mask to remove the values
        M = np.ma.masked_array(self.cumsum_mine.copy(), mask)
        # make the mask values = 0
        M[M.mask] = 0
        # find the max of the columns and return the sum
        N = np.amax(np.array(M), axis=-1)
        R = np.sum(N)
        return R

    # ========================  Class Mine  ==================================


def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of
    digging actions from the initial state of the mine.

    Return the sequence of actions, the final state and the payoff


    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state
    '''
    def recursive_search(state, act_func, pay_func, rec_func, state_tpl,
                         best_state, actions=None, is_3D=False):
        """
        This is the main recursive function for the DP process.
        :Param state: the current state as a tuple.
        :Param act_func: The lru_cache for the action function.
        :Param pay_func: The lru_cache for the payoff function.
        :Param rec_func: The lru_cache for the recursive function.
        :Param state_tpl: A tuple of currently visited states.
        :Param best_state: The best state as a tuple.
        :Param actions: A tuple of actions from the state or None.
        :Param is_3d: If we are checking 2D or 3D mine.
        """
        # Change to a list in order to be mutable
        state_lst = convert_to_list(state_tpl)

        # Get the state payoff and check it against the best
        state_payoff = pay_func(state)

        if state_payoff > payoff(best_state):
            best_state = state

        # Get a list of actions from this current state.
        given_state_actions = list(act_func(state))

        # Make current state an np array.
        state = np.array(state)
        if is_3D:
            key = str(state.flatten(order="C"))
        else:
            key = str(state)

        # Append the key to the state_lst
        state_lst.append(convert_to_tuple(key))

        if len(given_state_actions) > 0:
            for move in given_state_actions:
                new_state = state.copy()
                new_state[move] += 1
                if is_3D:
                    key = str(new_state.copy().flatten(order="C"))
                else:
                    key = str(new_state)

                # If this state currently not in the state_lst we visit it.
                if convert_to_tuple(key) not in state_lst:
                    best_state = recursive(convert_to_tuple(new_state),
                                           action,
                                           payoff,
                                           recursive,
                                           convert_to_tuple(state_lst),
                                           best_state,
                                           actions=move,
                                           is_3D=is_3D)
        return best_state

    # init vars.
    mine_dim = False if mine.is_2D else True
    best_state = ()
    state = mine.initial

    # init some lru_caches
    action = functools.lru_cache(maxsize=None)(mine.actions)
    payoff = functools.lru_cache(maxsize=None)(mine.payoff)
    recursive = functools.lru_cache(maxsize=None)(recursive_search)
    state_lst = []

    # run it
    best_state = recursive(state,
                           action,
                           payoff,
                           recursive,
                           convert_to_tuple(state_lst),
                           state,
                           is_3D=mine_dim)

    return mine.payoff(best_state), find_action_sequence(mine.initial, best_state), best_state


def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of
    digging actions from the initial state of the mine.


    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state
    '''
    # This is a small buffer for some negative dig cases
    BUFFER = np.abs(np.mean(mine.cumsum_mine))

    # setup a lru cache for the commonly called b and test function
    b = functools.lru_cache(maxsize=None)(mine.b)
    test = functools.lru_cache(maxsize=None)(mine.goal_test)

    # get the initial node and update our current highest
    node = search.Node(mine.initial)
    highest = test(node.state)

    # establish a priority queue.
    frontier = search.PriorityQueue(f=b)
    frontier.append(node)

    # As states can be revisited from other states we will explore like
    # a graph
    explored = set()
    while frontier:
        node = frontier.pop()
        # test the node and return the cost to of the current best state
        highest = test(node.state)

        # add the node to the explored list
        explored.add(node.state)

        # We will only search nodes which look to be optamistic
        for child in [c for c in node.expand(mine)
                      if (b(c) - BUFFER) < highest]:
            if child.state not in explored and child not in frontier:
                # The node child is considered "in frontier", if a node
                # already in frontier has the same state.
                # See PriortyQueue.__contains__()
                frontier.append(child)
            elif child in frontier:
                # A node in frontier has the same state as child
                # frontier[child] is the f-value of the node.
                # See method  PriorityQueue.__getitem__()
                if b(child) > frontier[child]:
                    # Replace the incumbent (that is the node
                    # already in the frontier) with child
                    del frontier[child]
                    frontier.append(child)
    return mine.payoff(mine.best), find_action_sequence(mine.initial, mine.best), mine.best


def find_action_sequence(s0, s1):
    """
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.

    Preconditions:
        s0 and s1 are legal states, s0<=s1 and

    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state

    Returns
    -------
    A sequence of actions to go from state s0 to state s1
    """
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]

    # If s0 and s1 is 2d tuple, it means the underground is 3d
    # Otherwise, s0 and s1 is 1d tuple.
    # Check legal of s0 < s1
    assert s0 <= s1

    path = []
    if s0 == s1:
        return []
    s0 = np.array(s0)
    s1 = np.array(s1)
    mask = np.full(s0.shape, True)
    MAX_POSITIVE_NUMBER = 99999999999
    while not (s0 == s1).all():
        loc = tuple(np.argwhere((s0 == np.min(s0,
                                              where=mask,
                                              initial=MAX_POSITIVE_NUMBER))
                                & mask)[0])

        if s0[loc] >= s1[loc]:
            mask[loc] = False
            continue

        s0[loc] += 1
        path.append(loc)
    return path
