#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:03:56 2019

@author: druee
"""

import unittest

import numpy as np

import readmet

##
#  optional debugging output
# import logging,sys
#  logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger()
# logger.level = logging.DEBUG
# stream_handler = logging.StreamHandler(sys.stdout)
#  logger.addHandler(stream_handler)


class Test_2D(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/so2-y00a.dmna')

    def test_vars(self):
        res = list(self.dmna.data.keys())
        self.assertEqual(res, ['con'])

    def test_shape(self):
        res = np.shape(self.dmna.data['con'])
        self.assertEqual(res, (100, 100, 1))

    def test_svalues(self):
        res = list(self.dmna.data['con'][80:91, 90, 0])
        cmp = [1.3, 1.3, 1.3, 1.3, 1.4, 1.3, 1.4, 1.3, 1.3, 1.4, 1.3]
        self.assertEqual(res, cmp)


class Test_3D(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/w1018a00.dmna')

    def test_vars(self):
        res = list(self.dmna.data.keys())
        self.assertEqual(res, ['Zp', 'Vx', 'Vy', 'Vs'])

    def test_shape(self):
        res = np.shape(self.dmna.data['Vs'])
        self.assertEqual(res, (101, 101, 20))

    def test_svalues(self):
        res = list(self.dmna.data['Vs'][40:51, 40, 3])
        cmp = [0.017965925857424736,
               0.020914768800139427,
               0.019798438996076584,
               0.021449146792292595,
               0.02623559907078743,
               0.027693092823028564,
               0.028620947152376175,
               0.033117726445198059,
               0.040409330278635025,
               0.046294711530208588,
               0.052625514566898346]
        self.assertAlmostEqual(res, cmp)


class Test_zeitreihe(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/zeitreihe.dmna')

    def test_vars(self):
        res = list(self.dmna.data.keys())
        self.assertEqual(set(res), set(['te', 'ra', 'ua', 'lm']))

    def test_shape(self):
        res = np.shape(self.dmna.data['ra'])
        self.assertEqual(res, (8784,))

    def test_svalues(self):
        res = list(self.dmna.data['ra'][80:91])
        cmp = [
            72.0,
            97.0,
            173.0,
            196.0,
            215.0,
            226.0,
            236.0,
            242.0,
            239.0,
            240.0,
            241.0]
        self.assertEqual(res, cmp)
