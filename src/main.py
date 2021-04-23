#!/usr/bin/env python3

"""
This is the file which will run the mining solutions
"""

from mining import my_team, Mine, search_dp_dig_plan, convert_to_tuple
from mining import search_bb_dig_plan, find_action_sequence

import numpy as np
import prac_mines as pm

import time


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


def testing_3D():
    quarry = Mine(pm.MINE_3D)
    quarry.console_display()
    print("initial")
    print(quarry.b(quarry.initial))
    print(quarry.payoff(quarry.initial))
    print()
    print("negative")
    print(quarry.b(((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0))))
    print(quarry.payoff(((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0))))
    print()
    print("best")
    print(quarry.b(((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))))
    print(quarry.payoff(((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))))
    print()
    print("just past best")
    print(quarry.b(((2, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1))))
    print(quarry.payoff(((2, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1))))
    print()
    print("edge")
    print(quarry.b(((1, 1, 1, 1), (1, 0, 0, 1), (0, 0, 0, 1))))
    print(quarry.payoff(((1, 1, 1, 1), (1, 0, 0, 1), (0, 0, 0, 1))))
    print()
    print("edge 2")
    print(quarry.b(((1, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))))
    print(quarry.payoff(((1, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))))
    # print(search_bb_dig_plan(quarry))


def testing_2D():
    quarry = Mine(pm.MINE_2D)
    quarry.console_display()
    print(quarry.b((0, 0, 0, 0, 0)))
    print(quarry.payoff((0, 0, 0, 0, 0)))
    print(quarry.b((3, 2, 2, 2, 1)))
    print(quarry.payoff((3, 2, 2, 2, 1)))
    print(quarry.b((3, 2, 3, 4, 3)))
    print(quarry.payoff((3, 2, 3, 4, 3)))
    print(quarry.b((2, 2, 3, 4, 3)))
    print(quarry.payoff((2, 2, 3, 4, 3)))
    print(quarry.b((4, 4, 4, 4, 4)))
    print(quarry.payoff((4, 4, 4, 4, 4)))
    print(search_bb_dig_plan(quarry))


def run_3D():
    quarry = Mine(pm.MINE_3D)
    quarry.console_display()
    tic = time.time()
    print(search_bb_dig_plan(quarry))
    toc = time.time()
    print("time:", toc-tic)


def run_2D():
    quarry = Mine(pm.MINE_2D)
    quarry.console_display()
    tic = time.time()
    print(search_bb_dig_plan(quarry))
    toc = time.time()
    print("time:", toc-tic)


if __name__ == "__main__":
    run_2D()
    run_3D()
