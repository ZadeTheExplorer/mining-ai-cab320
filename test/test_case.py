#!/usr/bin/env python3
"""
This is used to conduct a unit test on Mine
"""

from mining import my_team, Mine, search_dp_dig_plan
from mining import search_bb_dig_plan, find_action_sequence

import prac_mines as pm
import numpy as np

TEAM = [(9193243, 'Brodie', 'Smith'),
        (10250191, 'Keith', 'hall'),
        (10273913, 'Sy', 'Ha')]


class TestCase_2D:
    def test_team(self):
        assert my_team().sort() == TEAM.sort()

    def test_xdim(self):
        quarry = Mine(pm.CONTROL_1)
        assert quarry.len_x == pm.CONTROL_1.shape[0]

    def test_ydim(self):
        quarry = Mine(pm.CONTROL_1)
        assert quarry.len_y is None

    def test_zdim(self):
        quarry = Mine(pm.CONTROL_1)
        assert quarry.len_z == pm.CONTROL_1.shape[1]

    def test_cumsum(self):
        quarry = Mine(pm.CONTROL_1)
        assert np.array_equal(quarry.cumsum_mine,
                              pm.CONTROL_1_CUMSUM)

    def test_initial(self):
        quarry = Mine(pm.CONTROL_1)
        assert (len(np.array(quarry.initial).shape) ==
                (len(pm.CONTROL_1.shape) - 1))
        assert np.array(quarry.initial).shape[0] == pm.CONTROL_1.shape[0]

    def test_dangerous_2d(self):
        quarry = Mine(pm.CONTROL_1)
        for state, output in zip(pm.DANGEROUS_STATE,
                                 pm.DANGEROUS_VALUE):
            assert quarry.is_dangerous(state) is output

    def test_payoff_2d(self):
        # check control mine
        quarry = Mine(pm.CONTROL_1)
        for state, output in zip(pm.PAYOFF_STATE,
                                 pm.PAYOFF_VALUE):
            assert quarry.payoff(state) == output
        # check given mine
        quarry = Mine(pm.MINE_2D)
        assert (quarry.payoff(pm.MINE_2D_FINAL_STATE) ==
                pm.MINE_2D_PAYOFF)


class TestCase_3D:
    def test_team_3d(self):
        assert my_team().sort() == TEAM.sort()

    def test_xdim_3d(self):
        quarry = Mine(pm.CONTROL_2)
        assert quarry.len_x == pm.CONTROL_2.shape[0]

    def test_ydim_3d(self):
        quarry = Mine(pm.CONTROL_2)
        assert quarry.len_y is pm.CONTROL_2.shape[1]

    def test_zdim_3d(self):
        quarry = Mine(pm.CONTROL_2)
        assert quarry.len_z == pm.CONTROL_2.shape[2]

    def test_cumsum_3d(self):
        quarry = Mine(pm.CONTROL_2)
        assert (quarry.cumsum_mine.tolist().sort() ==
                pm.CONTROL_2_CUMSUM.tolist().sort())

    def test_initial(self):
        quarry = Mine(pm.CONTROL_2)
        assert (len(np.array(quarry.initial).shape) ==
                (len(pm.CONTROL_2.shape) - 1))
        assert np.array(quarry.initial).shape[0] == pm.CONTROL_2.shape[0]

    def test_dangerous_3d(self):
        quarry = Mine(pm.CONTROL_2)
        for state, output in zip(pm.DANGEROUS_3D_STATE,
                                 pm.DANGEROUS_3D_VALUE):
            assert quarry.is_dangerous(state) is output

    def test_payoff_3d(self):
        # check given mine
        quarry = Mine(pm.MINE_3D)
        assert (quarry.payoff(pm.MINE_3D_FINAL_STATE) ==
                pm.MINE_3D_PAYOFF)
