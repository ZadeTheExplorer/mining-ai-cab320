#!/usr/bin/env python3
"""
This is used to conduct a unit test on Mine
"""

from mining import my_team, Mine, search_dp_dig_plan
from mining import search_bb_dig_plan, find_action_sequence

import prac_mines
import numpy as np

TEAM = [(9193243, 'Brodie', 'Smith'),
        (10250191, 'Keith', 'hall'),
        (10273913, 'Sy', 'Ha')]


class TestCase:
    def test_team(self):
        assert my_team().sort() == TEAM.sort()

    def test_xdim(self):
        quarry = Mine(prac_mines.MINE_1)
        assert quarry.len_x == prac_mines.MINE_1.shape[0]

    def test_ydim(self):
        quarry = Mine(prac_mines.MINE_1)
        assert quarry.len_y is None

    def test_zdim(self):
        quarry = Mine(prac_mines.MINE_1)
        assert quarry.len_z == prac_mines.MINE_1.shape[1]

    def test_cumsum(self):
        quarry = Mine(prac_mines.MINE_1)
        assert np.array_equal(quarry.cumsum_mine,
                              prac_mines.MINE_1_CUMSUM)

    def test_initial(self):
        quarry = Mine(prac_mines.MINE_1)
        assert len(quarry.initial.shape) == (len(prac_mines.MINE_1.shape) - 1)
        assert quarry.initial.shape[0] == prac_mines.MINE_1.shape[0]

    def test_dangerous(self):
        quarry = Mine(prac_mines.MINE_1)
        assert quarry.is_dangerous(prac_mines.STATE_FAIL) is True
        assert quarry.is_dangerous(prac_mines.STATE_FAIL2) is True
        assert quarry.is_dangerous(prac_mines.STATE_PASS) is False
        assert quarry.is_dangerous(prac_mines.STATE_PASS2) is False
        assert quarry.is_dangerous(prac_mines.STATE_PASS3) is False
