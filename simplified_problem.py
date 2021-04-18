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
        return is_dangerous(updated, 1)

    # The generators which will return a valid action.
    if underground.ndim == 2:
        return ((x,) for x, z in zip(np.arange(len_x), state)
                if not pass_to(state, x))
    else:
        return ((x, y) for x, z in zip(np.arange(len_x), state[1])
                for y in np.arange(len(state[1])) if not
                pass_to(state, x, y))


def is_dangerous(state, tolerant):
    state = np.array(state)

    if any(np.greater(np.abs(np.diff(state)),
                          tolerant)):
        return True
    return False
    #print("This is x_diff = ",x_diff)

    #return x_diff.any()


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

def goal_state_3d(underground, tolerance=1):

    #state_product = product(range(0, underground.shape[1] + 1), repeat=underground.shape[0])

    # for state in state_product:
    #     print(state)

    len_z = underground.shape[0]
    len_x = underground.shape[1]
    len_y = underground.shape[2]

    state_tuple = []

    for x in range(len_x):
        y_tuple = (0,)*len_y
        state_tuple.append(y_tuple)
    
    print(state_tuple)

    #sp = product(range(0, (len_z + 1)), repeat=len_x*len_y)

    # for state in state_tuple:
    #     print(state)
    #     sp = list(product(range(0, len_z), repeat=len_y))
    #     sp2 = product(sp,sp)
    #     sp3 = product(sp2, sp)
        #print(sp)
    
    goal_sum = None
    best_payoff = None
    
    for state in state_tuple:
        
        if is_dangerous(state, tolerance):
            continue

        sum_state = payoff(underground, state)

        if best_payoff is None or sum_state > best_payoff:
            best_payoff = sum_state
            goal_sum = state


    return goal_sum, best_payoff

def goal_state_2d_with_memo(underground, tolerance, memo=None, finished=False):

    '''
    Problem Variables
        - goal_sum, best_payoff, is_dangerous(?)
    Recurrence Relation
        - If (L,P) is our payoff and column location, our possible moves are:
            - (L, P+P) if we don't change our column location 
            - (L-n, P+P) if we go back (relative term) a column location
            - (L+n, P+P) if we go forward (relative term) a column location
                - Where n is the distance the neighbour is away from the current location
    Base Cases
        - The column location/action isn't dangerous
        - The column location isn't shorter or longer than the depth of the column
    '''

    # Recursive Implementation
    #   Generate Actions
    #   If the payoffs of those actions are bigger, add them
    #   Generate Actions from those Actions
    #   Repeat

    # Declare payoffs
    best_state = ()
    best_payoff = 0
    action_list = []

    # Set up state for recursive implementation
    state = np.zeros(underground.shape[0])

    def recursive_search(underg, given_state, action=None):

        nonlocal best_payoff
        nonlocal best_state
        nonlocal action_list

        print("Given state = ",given_state)
        
        state_payoff = payoff(underg, given_state)

        if state_payoff > best_payoff:
            best_payoff = state_payoff
            best_state = given_state
        
        action_list.append(action)
        
        print("Best Payoff ", best_payoff)
        print("Best State ", best_state)
        print("Action List ", action_list)

        ### Version 1: Just gets the next available action

        if len(list(actions(underg,given_state))) > 0:

            next_action = next(actions(underg,given_state))

            print("Next Actions = ", list(actions(underg,given_state)))

            print("Next Action selected = ", next_action)

            new_state = given_state.copy()
            new_state[next_action] += 1

            recursive_search(underg, new_state, next_action)
        else:
            return
        
        ### Version 2: loops through every possible action
        # if len(list(actions(underg,given_state))) > 0:
        #     print("Actions Length ", len(list(actions(underg,given_state))))
        #     #for action in state_actions:
        #     for action in actions(underg, given_state):
        #         new_state = given_state.copy()
        #         new_state[action] += 1
        #         recursive_search(underg, new_state, action)
        # else:
        #     return
    
    recursive_search(underground, state)
    
    return best_state, best_payoff, action_list


def goal_state_2d(array, tolerant=1):
    """
    Find the max of selected
    :param array: 2d array representing for the underground
    :param tolerant:
    :return: goal state
    """
    # cumsum_mine_at_that_index = array.copy()
    # for i in range(0, array.shape[0]):
    #     for j in range(1, array.shape[1]):
    #         cumsum_mine_at_that_index[i, j] += cumsum_mine_at_that_index[i, j - 1]

    # state_product = cartesian_product(range(0, array.shape[1] + 1), array)
    # print("len", len(state_product))
    state_product = list(product(range(0, array.shape[1] + 1), repeat=array.shape[0]))
    

    goal_state = None
    max_sum = None
    actions_list = []
    previous_actions = []
    previous_sums = []
    num_actions = 0

    cumsum_mine = np.sum(array, axis=1 if array.ndim == 2 else 2)

    #print(cumsum_mine)

    for state in state_product:

        if is_dangerous(state, tolerant):
            continue

        state_actions = actions(array, state)

        list_actions = list(state_actions)
        num_actions += 1
        print("List Actions = ", list_actions, " Num = ",num_actions)

        # if list_actions not in previous_actions:
        #     previous_actions.append(list_actions)
        # else:
        #     #print("Previous Actions found = ", list_actions)
        #     continue

        
        sum_state = payoff(array, state)
        #print("Max Sum = ",max_sum, " Sum State = ", sum_state)

        if sum_state not in previous_sums:
            previous_sums.append(sum_state)
        else:
            #print("Previous State repeated = ",sum_state)
            continue

        if max_sum is not None:
            if sum_state < max_sum:
                continue

        if max_sum is None or sum_state > max_sum:
            max_sum = sum_state
            goal_state = state


        # #print(list_actions)

        # if len(list_actions) == 1:
        #     actions_list.append(list_actions)


    return goal_state, max_sum, actions_list


def test_1():
    # about ~0.5s
    array = np.random.uniform(low=-10, high=10, size=(5, 4))

    t0 = time.time()
    poss = goal_state_2d(array, 1)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)


def test_2():
    # about ~30s
    array = np.random.uniform(low=-10, high=10, size=(10, 4))

    t0 = time.time()
    poss = goal_state_2d(array, 1)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)


def test_3():
    # about ~30s
    array = np.random.uniform(low=-10, high=10, size=(5, 20))

    t0 = time.time()
    poss = goal_state_2d(array, 1)
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
    poss = goal_state_2d(array, 1)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d", poss)

def test_sanity_3d():

    array = np.array([[[ 0.455,  0.049,  2.38,   0.515],
                        [ 0.801, -0.09,  -1.815,  0.708],
                        [-0.857, -0.876, -1.936,  0.316]],
                        [[ 0.579,  1.311, -1.404, -0.236],
                        [ 0.072, -1.191, -0.839, -0.227],
                        [ 0.309,  1.188, -3.055,  0.97 ]],
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
    poss = goal_state_3d(array, 1)
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

    t0 = time.time()
    poss = goal_state_2d_with_memo(array, 1)
    t1 = time.time()
    print(f'The solver took {t1 - t0} seconds')
    print(array)
    print("goal_state_2d_with_memo", poss)


if __name__ == '__main__':
    # The larger mine (larger x), the crazier the time complexity comes
    #test_sanity()
    #test_sanity_3d()
    #test_1()
    #test_2()
    #test_3()
    test_memo_2d()
