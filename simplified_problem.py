# Find out the best result  (sum of the matrix)
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time


def actions(underground, state):
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


    len_x = underground.shape[0]
    len_z = (underground.shape[1] if underground.ndim == 2
                    else underground.shape[2])
    len_y = underground.shape[1] if underground.ndim == 3 else None

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
        if y is None:  # Its a 1D state we are testing.
            np.add.at(updated, x, 1)
            if updated[x] > len_z:
                return True
        else:
            np.add.at(updated, (x, y), 1)
            if updated[x, y] > len_z:
                return True
        # Else test if it is dangerous.
        return is_dangerous(updated, underground, 1)

    # The generators which will return a valid action.
    if underground.ndim == 2:
        return ((x,) for x, z in zip(np.arange(len_x), state)
                if not pass_to(state, x))
    else:
        return ((x, y) for x, z in zip(np.arange(len_x), state[1])
                for y in np.arange(len(state[1])) if not
                pass_to(state, x, y))


def is_dangerous(state, underground, tolerance):
    state = np.array(state)

    # print("State ", state)
    # print("np.diff ", np.diff(state))
    # print("np.abs ", np.abs(np.diff(state)))
    # print("np.greater ", np.greater(np.abs(np.diff(state)),
    #                       tolerant))
    # for value in np.greater(np.abs(np.diff(state)),
    #                       tolerant):
        
    #     if (value == True).any():
    #         return True
    #     else:
    #         return False

    # if any(np.greater(np.abs(np.diff(state)),
    #                       tolerant)):
    #     return True
    # return False

    state = np.array(state)

    # get the diff along the x dim
    x_diff = np.greater(np.abs(np.diff(state, axis=0)),
                        tolerance)
    if x_diff.any():
        return True

    # if we have a 3d underground
    if underground.ndim == 3:
        # get the diff along the y dim
        y_diff = np.greater(np.abs(np.diff(state, axis=1)),
                            tolerance)
        if y_diff.any():
            return True
        # now work out the diff for the diagonals
        diag_1 = np.greater(np.abs((state[:-1, :-1] - state[1:, 1:])),
                            tolerance)
        n_state = np.rot90(state)
        diag_2 = np.greater(np.abs((n_state[:-1, :-1] - n_state[1:, 1:])),
                            tolerance)
        if diag_1.any() or diag_2.any():
            return True
    return False


def payoff(array, state):
    state = np.array(state)

    if array.ndim == 2:
        # Create a mask using the state and then sum
        mask = state[:, None] > np.arange(array.shape[1])
        return np.sum(array, where=mask)
    else:
        # As above
        mask = state[:, :, None] > np.arange(array.shape[2])
        return np.sum(array, where=mask)


def cartesian_product(arr1, arr2):
    # return the list of all the computed tuple
    # using the product() method
    return list(product(arr1, repeat=arr2.shape[0]))

def goal_state_3d_with_memo(underground):

    # Declare return values
    best_state = ()
    best_payoff = 0
    action_list = []

    underg1 = underground.flatten(order="C")

    underground_correct = underg1.reshape(3,4,5)

    cumsum_mine = np.sum(underground,
                                  axis=1 if underground.ndim == 2 else 2)
    
    #print(cumsum_mine)

    state = np.zeros((underground_correct.shape[0],underground_correct.shape[1]))

    whole_memo = dict()

    def recursive_search(underg, given_state, action=None, memo=None):

        nonlocal best_payoff
        nonlocal best_state
        nonlocal action_list
        nonlocal whole_memo

        state_payoff = payoff(underg, given_state)

        #print("Given_state ", given_state)

        if state_payoff > best_payoff:
            best_payoff = state_payoff
            best_state = given_state

        #print("State Payoff ", state_payoff)
        #print("Best Payoff ", best_payoff)
        #print("Best State ", best_state)

        given_state_actions = list(actions(underg, given_state))

        whole_memo[str(given_state.flatten(order="C"))] = (given_state_actions, state_payoff)

        print("Memo Entry Payoff for ", str(given_state.flatten(order="C")), " ",whole_memo[str(given_state.flatten(order="C"))][1])

        if len(given_state_actions) > 0:

            for action in given_state_actions:

                new_state = given_state.copy()
                new_state[action] += 1

                #print("String New State ", str(new_state.flatten(order="C")))

                if str(new_state.flatten(order="C")) not in whole_memo:
                    recursive_search(underg, new_state, action, whole_memo)
                        
        else:
            return

    recursive_search(underground_correct, state, whole_memo)


    '''
    Use Patrick's find_action_sequence here with best_state
    '''
    return best_payoff, action_list, best_state

def goal_state_2d_with_memo(underground):

    # Declare return values
    best_state = ()
    best_payoff = 0
    action_list = []

    cumsum_mine = np.sum(underground,
                                  axis=1 if underground.ndim == 2 else 2)
    
    #print(cumsum_mine)

    # Set up state for recursive implementation
    state = np.zeros(underground.shape[0])

    whole_memo = dict()

    def recursive_search(underg, given_state, action=None, memo=None):

        '''
        Step 1 - Calculate the actions and payoff for given state
        Step 1A - Check if payoff is better than previous
                - If it is, store payoff as new best payoff, and state as best_state
        Step 2 - Store those in the memo
        Step 3 - For every possible action, check if its state is in the memo
                - If it is, skip it/don't do it
                - If it is not, call recursive search with new state and memo

        '''

        nonlocal best_payoff
        nonlocal best_state
        nonlocal action_list
        nonlocal whole_memo

        state_payoff = payoff(underg, given_state)

        if state_payoff > best_payoff:
            best_payoff = state_payoff
            best_state = tuple(given_state)

        given_state_actions = list(actions(underg, given_state))

        whole_memo[str(given_state)] = (given_state_actions, state_payoff)
        
        if len(given_state_actions) > 0:

            for action in given_state_actions:

                new_state = given_state.copy()
                new_state[action] += 1

                if str(new_state) not in whole_memo:
                    recursive_search(underg, new_state, action, whole_memo)
                        
        else:
            return

    recursive_search(underground, state, whole_memo)

    '''
    Use Patrick's find_action_sequence here with best_state
    '''
    return best_payoff, action_list, best_state


def test_1():
    # about ~0.5s
    array = np.random.uniform(low=-10, high=10, size=(5, 4))

    t0 = time.time()
    poss = goal_state_2d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)


def test_2():
    # about ~30s
    array = np.random.uniform(low=-10, high=10, size=(10, 4))

    t0 = time.time()
    poss = goal_state_2d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)


def test_3():
    # about ~30s
    array = np.random.uniform(low=-10, high=10, size=(5, 20))

    t0 = time.time()
    poss = goal_state_2d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)


def test_sanity():
    # about ~0.02s
    array = np.array([[-0.814, 0.637, 1.824, -0.563],
                      [0.559, -0.234, -0.366, 0.07],
                      [0.175, -0.284, 0.026, -0.316],
                      [0.212, 0.088, 0.304, 0.604],
                      [-1.231, 1.558, -0.467, -0.371]])

    t0 = time.time()
    poss = goal_state_2d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)

def test_sanity_3d():

    array = np.array([ [ [ 0.455,  0.049,  2.38,   0.515], [ 0.801, -0.09,  -1.815,  0.708], [-0.857, -0.876, -1.936,  0.316] ],
                        [ [ 0.579,  1.311, -1.404, -0.236], [ 0.072, -1.191, -0.839, -0.227], [ 0.309,  1.188, -3.055,  0.97 ] ],
                        [[-0.54,  -0.061,  1.518, -0.466],
                        [-2.183, -1.083,  0.457,  0.874],
                        [-1.623, -0.16,  -0.535,  1.097]],
                        [[-0.995,  0.185, -0.856, -1.241],
                        [ 0.858,  0.78,  -1.029,  1.563],
                        [ 0.364,  0.888, -1.561,  0.234]],
                        [[-0.771, -1.959,  0.658, -0.354],
                        [-1.504, -0.763,  0.915, -2.284],
                        [ 0.097, -0.546, -1.992, -0.296]]])

    t0 = time.time()
    poss = goal_state_3d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_3d", poss)

def test_memo_2d():
    array = np.array([[-0.814, 0.637, 1.824, -0.563],
                      [0.559, -0.234, -0.366, 0.07],
                      [0.175, -0.284, 0.026, -0.316],
                      [0.212, 0.088, 0.304, 0.604],
                      [-1.231, 1.558, -0.467, -0.371]])

    #array = np.random.uniform(low=-10, high=10, size=(10, 4))
    #array = np.random.uniform(low=-10, high=10, size=(5, 20))

    t0 = time.time()
    poss = goal_state_2d_with_memo(array)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d_with_memo", poss)


if __name__ == '__main__':
    # The larger mine (larger x), the crazier the time complexity comes
    #test_sanity()
    test_sanity_3d()
    #test_1()
    #test_2()
    #test_3()
    #test_memo_2d()
