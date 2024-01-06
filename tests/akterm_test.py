#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:03:56 2021

@author: druee
"""

import unittest
# import unittest.mock as mock

import filecmp
import os
import sys
import tempfile
import pandas as pd
import numpy as np

import readmet


# # optional debugging output
if True:
    import logging
    logger = logging.getLogger()
    logger.level = logging.DEBUG
    stream_handler = logging.StreamHandler(sys.stdout)
else:
    logging = None

akterm_test_file = 'tests/testfile.akterm'
aktermx_test_file = 'tests/testfile_extended.akterm'

_AKT_COLUMNS = ['KENN', 'STA', 'JAHR', 'MON', 'TAG', 'STUN', 'NULL',
                'QDD', 'QFF', 'DD', 'FF', 'QQ1', 'KM', 'QQ2', 'HM', 'QQ3']
_AKTN_COLUMNS = _AKT_COLUMNS + ['PP', 'QPP']
_PREC_KEYWORD = 'Niederschlag'


class Test_load_header(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(akterm_test_file)
        self.header = self.akterm.header
        self.heights = self.akterm.heights

    def test_header_type(self):
        self.assertTrue(isinstance(self.header, list))

    def test_header_line_type(self):
        self.assertTrue(
            all([isinstance(x, str) for x in self.header]))

    def test_header_comment_char_removed(self):
        self.assertTrue(
            not any([x.startswith('*') for x in self.header]))

    def test_anemometerheights_type(self):
        self.assertTrue(isinstance(self.heights, np.ndarray))

    def test_anemometerheights_value_type(self):
        self.assertTrue(
            all([isinstance(x, np.floating) for x in self.heights]))

    def test_header_lines(self):
        reference = [
            "AKTERM-Zeitreihe, Test-Datei",
            "Zeitraum 01/2000 bis 12/2000",
            "anonymisierte Daten, generiert: 17.01.2021"]
        self.assertListEqual(self.header, reference)

    def test_anemometerheights_value(self):
        reference = [8.5, 10.0, 12.4, 14.7, 17.6, 22.6, 28.0, 32.1, 35.5]
        np.testing.assert_almost_equal(
            list(self.heights), reference, decimal=1)

    def test_akterm_extended(self):
        self.assertFalse(self.akterm.prec)


class Test_load_headerx(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(aktermx_test_file)
        self.header = self.akterm.header
        self.heights = self.akterm.heights

    def test_header_type(self):
        self.assertTrue(isinstance(self.header, list))

    def test_header_line_type(self):
        self.assertTrue(
            all([isinstance(x, str) for x in self.header]))

    def test_header_comment_char_removed(self):
        self.assertTrue(
            not any([x.startswith('*') for x in self.header]))

    def test_anemometerheights_type(self):
        self.assertTrue(isinstance(self.heights, np.ndarray))

    def test_anemometerheights_value_type(self):
        self.assertTrue(
            all([isinstance(x, np.floating) for x in self.heights]))

    def test_header_lines(self):
        reference = [
            "AKTERM-Zeitreihe, Test-Datei mit Niederschlag",
            "Zeitraum 01/2000 bis 12/2000",
            "anonymisierte Daten, generiert: 26.11.2022"]
        self.assertListEqual(self.header, reference)

    def test_anemometerheights_value(self):
        reference = [0.8, 1.3, 2.3, 3.5, 5.6, 10.5, 17.2, 23.2, 28.7]
        np.testing.assert_almost_equal(
            list(self.heights), reference, decimal=1)

    def test_akterm_extended(self):
        self.assertTrue(self.akterm.prec)

    def test_header_keyword(self):
        self.assertTrue(_PREC_KEYWORD in self.header[0])


class Test_load_data(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(akterm_test_file)

    def test_data_type(self):
        self.assertTrue(isinstance(self.akterm.data, pd.DataFrame))

    def test_data_length(self):
        self.assertEqual(len(self.akterm.data), 8784)

    def test_data_columns(self):
        self.assertListEqual(_AKT_COLUMNS, list(self.akterm.data.columns))

    def test_data_datetime(self):
        self.assertTrue(
            all(pd.to_datetime("2000-01-01") <= self.akterm.data.index) and
            all(self.akterm.data.index < pd.to_datetime("2001-01-01"))
            )

    def test_data_km(self):
        self.assertTrue(
            all(x in [1, 2, 3, 4, 5, 6, 9] for x in self.akterm.data['KM'])
            )

    def text_data_h_anemo_invalid(self):
        self.assertRaises(ValueError, self.akterm.get_h_anemo(1.1))
        self.assertRaises(ValueError, self.akterm.get_h_anemo(-1.0))

    def text_data_h_anemo(self):
        self.assertAlmostEqual(self.akterm.get_h_anemo(0.5), 22.6)


class Test_load_datax(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(aktermx_test_file)

    def test_data_type(self):
        self.assertTrue(isinstance(self.akterm.data, pd.DataFrame))

    def test_data_length(self):
        self.assertEqual(len(self.akterm.data), 8784)

    def test_data_columns(self):
        self.assertListEqual(_AKTN_COLUMNS, list(self.akterm.data.columns))

    def test_data_datetime(self):
        self.assertTrue(
            all(pd.to_datetime("2000-01-01") <= self.akterm.data.index) and
            all(self.akterm.data.index < pd.to_datetime("2001-01-01"))
            )

    def text_data_h_anemo_invalid(self):
        self.assertRaises(ValueError, self.akterm.get_h_anemo(1.1))
        self.assertRaises(ValueError, self.akterm.get_h_anemo(-1.0))

    def text_data_h_anemo(self):
        self.assertAlmostEqual(self.akterm.get_h_anemo(0.5), 10.5)


class Test_load_write(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(akterm_test_file)
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, 'wite_test.akterm')

    def test_write_succeeds(self):
        self.akterm.write(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        os.remove(self.filename)

    def test_write_back(self):
        self.akterm.write(self.filename)
        self.assertTrue(filecmp.cmp(akterm_test_file, self.filename))
        os.remove(self.filename)


class Test_load_writex(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        self.akterm = readmet.akterm.DataFile(aktermx_test_file)
        self.tempdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tempdir, 'wite_testx.akterm')

    def test_write_succeeds(self):
        self.akterm.write(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        os.remove(self.filename)

    def test_write_back(self):
        self.akterm.write(self.filename)
        if logging is not None:
            with open(aktermx_test_file, 'r') as file1:
                with open(self.filename, 'r') as file2:
                    for line1, line2 in zip(
                            file1.readlines(),
                            file2.readlines()):
                        if line1 != line2:
                            #logging.debug('<'+line1)
                            #logging.debug('>'+line2)
                            print('<' + line1)
                            print('>' + line2)
        self.assertTrue(filecmp.cmp(aktermx_test_file, self.filename))
        os.remove(self.filename)
