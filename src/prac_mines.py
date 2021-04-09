#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A common place to store some mines for testing.
"""

import numpy as np

# place holder

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

STATE_FAIL = (0, 0, 0, 0, 3, 0, 0, 0, 0)
# This should be an impossible edge case
STATE_FAIL2 = (5, 3, 2, 1, 0, 0, 0, 0, 0)
STATE_PASS = (0, 0, 0, 0, 1, 2, 1, 0, 0)
STATE_PASS2 = (0, 1, 2, 3, 4, 3, 2, 1, 0)
STATE_PASS3 = (4, 3, 2, 1, 0, 0, 0, 0, 0)
