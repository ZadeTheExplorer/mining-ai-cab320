#!/usr/bin/env python3
"""

"""

from mining import my_team, Mine, search_dp_dig_plan
from mining import search_bb_dig_plan, find_action_sequence

import numpy as np

MINE_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4, 4, 4, 4],
                   [3, 3, 3, 3, 3, 3, 3, 3],
                   [2, 2, 2, 2, 2, 2, 2, 2],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0]])

MINE_1_CUMSUM = np.array([0, 8, 16, 24, 32, 24, 16, 8, 0])

TEAM = [(9193243, 'Brodie', 'Smith'),
        (10250191, 'Keith', 'hall'),
        (10273913, 'Sy', 'Ha')]


class TestCase:
    def test_team(self):
        assert my_team().sort() == TEAM.sort()

    def test_xdim(self):
        quarry = Mine(MINE_1)
        assert quarry.len_x == MINE_1.shape[0]

    def test_ydim(self):
        quarry = Mine(MINE_1)
        assert quarry.len_y is None

    def test_zdim(self):
        quarry = Mine(MINE_1)
        assert quarry.len_z == MINE_1.shape[1]

    def test_cumsum(self):
        quarry = Mine(MINE_1)
        assert np.array_equal(quarry.cumsum_mine, MINE_1_CUMSUM)

    def test_initial(self):
        quarry = Mine(MINE_1)
        assert quarry.initial.shape == MINE_1.shape
