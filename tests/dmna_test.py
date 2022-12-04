#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:03:56 2019

@author: druee
"""
import os
import unittest

import numpy as np

import readmet

#
# optional debugging output
import logging,sys
logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)


class Test_read_2D(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/so2-y00a.dmna')

    def test_vars(self):
        res = self.dmna.variables
        self.assertEqual(res, ['con'])

    def test_shape(self):
        res = np.shape(self.dmna.data['con'])
        self.assertEqual(res, (100, 100, 1))

    def test_svalues(self):
        res = list(self.dmna.data['con'][40:60,50,0])
        cmp = [1.5, 2., 2.4, 2.9, 3.7, 5.2, 6.6, 9.9, 20.7,
               84.5, 202.2, 118.9, 65.4, 42.2, 29.5, 21.3, 16.3, 13.1,
               10., 8.4]
        self.assertEqual(res, cmp)


class Test_read_3D(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/w1018a00.dmna')

    def test_vars(self):
        res = self.dmna.variables
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
        logging.debug(str(res))
        np.testing.assert_almost_equal(res, cmp, decimal=3)


class Test_read_zeitreihe(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.dmna = readmet.dmna.DataFile('tests/zeitreihe.dmna')

    def test_vars(self):
        res = self.dmna.variables
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

class Test_write_2D(unittest.TestCase):
    def setup_method(self, method):
        self.testfile = 'tests/write.dmna'
        self.original = readmet.dmna.DataFile('tests/so2-y00a.dmna')
        self.original.write(self.testfile)
        self.reread = readmet.dmna.DataFile(self.testfile)

    def test_vars(self):
        ref = self.original.variables
        res = self.reread.variables
        self.assertEqual(res, ref)

    def test_shape(self):
        ref = np.shape(self.original.data['con'])
        res = np.shape(self.reread.data['con'])
        self.assertEqual(res, ref)

    def test_svalues(self):
        ref = list(self.original.data['con'][40:60,50,0])
        res = list(self.reread.data['con'][40:60,50,0])
        self.assertEqual(res, ref)

    def teardown_method(self, method):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

class Test_write_3D(unittest.TestCase):

    def setup_method(self, method):
        self.testfile = 'tests/write.dmna'
        self.original = readmet.dmna.DataFile('tests/w1018a00.dmna')
        self.original.write(self.testfile)
        self.reread = readmet.dmna.DataFile(self.testfile)

    def test_vars(self):
        ref = self.original.variables
        res = self.reread.variables
        self.assertEqual(res, ref)

    def test_shape(self):
        ref = np.shape(self.original.data['Vs'])
        res = np.shape(self.reread.data['Vs'])
        self.assertEqual(res, ref)

    def test_svalues(self):
        ref = list(self.original.data['Vs'][40:51, 40, 3])
        res = list(self.reread.data['Vs'][40:51, 40, 3])
        np.testing.assert_almost_equal(res, ref)

    def teardown_method(self, method):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)


class Test_write_zeitreihe(unittest.TestCase):
    def setup_method(self, method):
        self.testfile = 'tests/write.dmna'
        self.original = readmet.dmna.DataFile('tests/zeitreihe.dmna')
        self.original.write(self.testfile)
        self.reread = readmet.dmna.DataFile(self.testfile)

    def test_vars(self):
        ref = self.original.variables
        res = self.reread.variables
        self.assertEqual(set(res), set(ref))

    def test_shape(self):
        ref = np.shape(self.original.data['ra'])
        res = np.shape(self.reread.data['ra'])
        self.assertEqual(res, ref)

    def test_svalues(self):
        ref = list(self.original.data['ra'][80:91])
        res = list(self.reread.data['ra'][80:91])
        self.assertEqual(res, ref)

    def teardown_method(self, method):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)


