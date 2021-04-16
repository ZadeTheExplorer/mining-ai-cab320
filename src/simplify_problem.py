# Find out the best result  (sum of the matrix)
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time


def is_dangerous(state, tolerant):
    state = np.array(state)

    x_diff = np.greater(np.abs(np.diff(state, axis=0)), tolerant)
    return x_diff.any()


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
    state_product = product(range(0, array.shape[1] + 1), repeat=array.shape[0])
    goal_state = None
    max_sum = None
    for state in state_product:
        if is_dangerous(state, tolerant):
            continue

        sum_state = payoff(array, state)

        if max_sum is None or sum_state > max_sum:
            max_sum = sum_state
            goal_state = state

    return goal_state, max_sum


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


if __name__ == '__main__':
    # The larger mine (larger x), the crazier the time complexity comes
    test_sanity()
