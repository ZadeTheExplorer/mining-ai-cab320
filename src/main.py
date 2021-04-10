#!/usr/bin/env python3

"""
This is the file which will run the mining solutions
"""

from mining import my_team, Mine, search_dp_dig_plan
from mining import search_bb_dig_plan, find_action_sequence

import numpy as np
import prac_mines as pm


def return_mine(third=False, shape=None, style=None):
    """
    This is a function that will provide random mines of different styles.
    :Param thirminimad {boolean}: If we require a (x, y, z) array.
    :Param shape {tuple x3}: A tuple of the desired mine dimentions.
    :Param style {string}: Used to select a desired style if any.
    """
    if shape is None:
        shape = random_shape(third)

    if style is None:
        return np.random.randint(-10, 10, shape)
    elif style == "v":
        return v_array(shape)
    elif style == "o":
        return o_array(shape)


def v_array(shape):
    """

    """
    part1_col = np.arange(np.ceil(shape[0]/2))
    part2_col = np.flip(np.arange(np.floor(shape[0]/2)))
    column = np.append(part1_col, part2_col)
    array = np.tile(column, (shape[-1], 1))
    return np.transpose(array)


def o_array(shape):
    """

    """
    pass


def random_shape(third):
    """

    """
    if third is False:
        return (np.random.randint(2, 10),
                np.random.randint(2, 10))
    return (np.random.randint(2, 10),
            np.random.randint(2, 10),
            np.random.randint(2, 10))


def sumrowsby_index_v2(a, index):
    lens = np.array([i for i in index])
    id_ar = np.zeros((len(lens), a.shape[0]))
    c = np.concatenate(index)
    r = np.repeat(np.arange(len(index)), lens)
    id_ar[r, c] = 1
    return id_ar.dot(a)


if __name__ == "__main__":
    # underground = return_mine(style="v")
    Quarry = Mine(pm.MINE_2D)
    Quarry.console_display()
    print(Quarry.payoff(pm.MINE_2D_FINAL_STATE))
