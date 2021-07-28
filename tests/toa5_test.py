#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:03:56 2021

@author: druee
"""

import unittest
import unittest.mock as mock

import os
import tempfile
import pandas as pd

import readmet

##
# optional debugging output
#import logging,sys
# logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger()
#logger.level = logging.DEBUG
#stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)

toa5_test_file = 'tests/TOA5_CR3000.data_2015_06_17_0010.dat'


class Test_read_header(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.header = readmet.toa5.get_header(toa5_test_file)

    def test_type(self):
        self.assertTrue(isinstance(self.header, readmet.toa5.Header))

    def test_line1(self):
        reference = {'station_name': "TEST_SITE",
                     'logger_name': "CR3000",
                     'logger_serial': "1234",
                     'logger_os': "CR3000.Std.28",
                     'logger_prog': "CPU:TESTPROG.CR3",
                     'logger_sig': "57003",
                     'table_name': "test_data",
                     }
        res = {k:v for k,v in self.header.items() if k in reference.keys()}
        self.assertDictEqual(res, reference)

    def test_line2(self):
        reference = ["TIMESTAMP", "RECORD", "AirTC_Avg", "RH_Avg", "Batt_Volt_Avg",
                     "BP_mbar_Avg", "h2o_Avg", "co2_Avg", "Ts_Avg", "Ux_Avg", "Uy_Avg", "Uz_Avg"]
        self.assertListEqual(self.header['column_names'], reference)

    def test_line3(self):
        reference = ["TS", "RN", "Deg C", "%", "Volts", "mbar",
                     "g/m^3", "mg/m^3", "C", "m/s", "m/s", "m/s"]
        self.assertListEqual(self.header['column_units'], reference)

    def test_line4(self):
        reference = ["", "", "Avg", "Avg", "Avg", "Avg",
                     "Avg", "Avg", "Avg", "Avg", "Avg", "Avg"]
        self.assertListEqual(self.header['column_sampling'], reference)


class Test_write_header(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, 'TOA5_test.dat')
        self.header = readmet.toa5.get_header(toa5_test_file)

    def test_write(self):
        readmet.toa5.write_header(self.filename, self.header)
        self.assertTrue(os.path.exists(self.filename))
        os.remove(self.filename)

    def test_read_back(self):
        readmet.toa5.write_header(self.filename, self.header)
        res = readmet.toa5.get_header(self.filename)
        self.assertDictEqual(res, self.header)
        os.remove(self.filename)


class Test_check_file(unittest.TestCase):
    # return values:
    #   1: file is valid TOA5
    #   0: file is not TOA5
    #  -1: file not found
    #   2: read error
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.tempdir = tempfile.mkdtemp()

    def test_file_ok(self):
        test_file = os.path.join(self.tempdir, 'TOA5_file_ok.dat')
        with open(toa5_test_file, 'r') as f:
            with open(test_file, 'w') as F:
                for r in f.readlines():
                    F.write(r)
        res = readmet.toa5.check_file(test_file)
        self.assertEqual(res, 1)
        os.remove(test_file)

    def test_file_invalid(self):
        test_file = os.path.join(self.tempdir, 'TOA5_file_ok.dat')
        l = 0
        with open(toa5_test_file, 'r') as f:
            with open(test_file, 'w') as F:
                for r in f.readlines():
                    if l >= 3:
                        F.write(r)
                    l = l + 1
        res = readmet.toa5.check_file(test_file)
        self.assertEqual(res, 0)
        os.remove(test_file)

    def test_file_nofound(self):
        test_file = os.path.join(self.tempdir, 'blah.dat')
        res = readmet.toa5.check_file(test_file)
        self.assertEqual(res, -1)

    def test_file_err(self):
        test_file = os.path.join(self.tempdir, 'TOA5_file_ok.dat')
        with open(test_file, 'wb') as F:
            F.write(b'')
        with mock.patch("builtins.open",
                        side_effect=IOError('IO-Error')) as mock_io:
            assert readmet.toa5.check_file(test_file) == 2
        os.remove(test_file)

class Test_read_data(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.data = readmet.toa5.get_data(toa5_test_file)

    def test_type(self):
        self.assertEqual(len(self.data.index), 144)

    def test_time(self):
        ref = list(pd.to_datetime(["2015-06-17 00:10:00",
                                   "2015-06-18 00:00:00"]))
        res = list([self.data.index[0], self.data.index[-1]])
        self.assertListEqual(res, ref)

    def test_value1(self):
        ref = 596.94072708
        res = self.data['BP_mbar_Avg'].mean()
        self.assertAlmostEqual(res, ref, places=2)

    def test_value2(self):
        ref = -0.033878819433333
        res = self.data['Uz_Avg'].mean()
        self.assertAlmostEqual(res, ref, places=5)

class Test_write_data(unittest.TestCase):
    # no speretae tests needed
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)

    def test_dummy(self):
        pass


class Test_read_file(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.header, self.data = readmet.toa5.read(toa5_test_file)

    def test_type(self):
        self.assertTrue(isinstance(self.header, readmet.toa5.Header))
        self.assertTrue(isinstance(self.data, pd.DataFrame))

    def test_consistency(self):
        self.assertListEqual(self.header['column_names'],
                             list(self.data.columns))

class Test_write_file(unittest.TestCase):
    def assertDataframeEqual(self, a, b):
        try:
            pd.util.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(e.args) from e

    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, 'TOA5_test.dat')
        self.header, self.data = readmet.toa5.read(toa5_test_file)

    def test_write(self):
        readmet.toa5.write_file(self.filename, self.header, self.data)
        self.assertTrue(os.path.exists(self.filename))
        os.remove(self.filename)

    def test_read_back(self):
        readmet.toa5.write_file(self.filename, self.header, self.data)
        res_head, res_data = readmet.toa5.read(self.filename)
        self.assertDictEqual(res_head, self.header)
        self.assertDataframeEqual(res_data, self.data)
        os.remove(self.filename)
