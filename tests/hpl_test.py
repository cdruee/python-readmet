#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for :mod:`readmet.hpl`.

Uses the two sample files shipped alongside this test (both taken
from a real Halo Photonics wind-lidar instrument):

* ``Wind_Profile_00_20260919_121707.hpl`` -- a regular scan file
  (6 rays, 750 gates each, "Wind profile - overlapping").
* ``Processed_Wind_Profile_77_20260919_121707.hpl`` -- the header-less
  "Processed Wind Profile" variant (750 levels).

A handful of additional, deliberately minimal ``.hpl`` files are
synthesized on the fly (see ``_write_minimal_file``) to exercise edge
cases the two real sample files don't cover on their own, and to pin
down specific bugs found and fixed during development (see the
"regression tests" section).
"""
from pathlib import Path

import pandas as pd
import pytest

from readmet import hpl

TESTDIR = Path(__file__).parent
WIND_PROFILE_FILE = TESTDIR / 'Wind_Profile_00_20260919_121707.hpl'
PROCESSED_PROFILE_FILE = (
    TESTDIR / 'Processed_Wind_Profile_77_20260919_121707.hpl')


# =========================================================================
# regular scan file: Wind_Profile_00_20260919_121707.hpl
# =========================================================================

class TestRegularScanFile:

    @pytest.fixture
    def datafile(self):
        return hpl.DataFile(file=str(WIND_PROFILE_FILE))

    def test_header_basics(self, datafile):
        assert datafile.header['instrument'] == '77'
        assert datafile.header['gates'] == '750'
        assert datafile.header['rays'] == '6'
        assert datafile.header['format_1'] == 'f9.6,1x,f6.2,1x,f6.2'
        assert datafile.header['format_2'] == (
            'i3,1x,f6.4,1x,f8.6,1x,e12.6 - repeat for no. gates')

    def test_type_and_mode(self, datafile):
        assert datafile.type == 'Wind profile'
        assert datafile.mode == 'overlapping'

    def test_timestamp(self, datafile):
        assert datafile.timestamp == pd.Timestamp(
            '2026-09-19 12:17:16.050000')

    def test_vars(self, datafile):
        assert datafile.vars['type'].tolist() == ['i', 'f', 'f', 'f']
        assert datafile.vars.index.tolist() == [
            'Range Gate', 'Doppler', 'Intensity', 'Beta']

    def test_rays_count_and_gates(self, datafile):
        assert len(datafile.rays) == 6
        assert all(r.gates == 750 for r in datafile.rays)

    def test_ray_angles(self, datafile):
        # regression test: a walrus/`.index()` precedence bug used to
        # make every ray silently read the same value here
        azimuths = [r.azimuth for r in datafile.rays]
        assert azimuths == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        assert all(r.elevation == 85.0 for r in datafile.rays)
        assert all(r.pitch == 0.01 for r in datafile.rays)

    def test_ray_times_are_increasing(self, datafile):
        times = [r.time for r in datafile.rays]
        assert times == sorted(times)
        assert times[0] == pd.Timestamp('2026-09-19 12:17:14.979984')

    def test_ray_data_columns(self, datafile):
        for r in datafile.rays:
            assert r.data.columns.tolist() == [
                'Range Gate', 'Doppler', 'Intensity', 'Beta']
            assert len(r.data.index) == 750

    def test_ray_data_values(self, datafile):
        first = datafile.rays[0].data
        assert first['Range Gate'].iloc[0] == '0'
        assert float(first['Doppler'].iloc[0]) == pytest.approx(0.8026)
        assert float(first['Intensity'].iloc[0]) == pytest.approx(1.003260)

    def test_profile_is_none_for_regular_file(self, datafile):
        assert datafile.profile is None

    def test_text_discarded_by_default(self):
        df = hpl.DataFile(file=str(WIND_PROFILE_FILE))
        assert df.text is None

    def test_text_retained_when_requested(self):
        df = hpl.DataFile(file=str(WIND_PROFILE_FILE), text=True)
        assert isinstance(df.text, list)
        assert df.text[0].startswith('Filename')


# =========================================================================
# DataFile.filter()
# =========================================================================

class TestFilter:

    @pytest.fixture
    def datafile(self):
        return hpl.DataFile(file=str(WIND_PROFILE_FILE))

    def test_no_filter_returns_all_as_list(self, datafile):
        result = datafile.filter()
        assert isinstance(result, list)
        assert len(result) == 6

    def test_scalar_number(self, datafile):
        # regression test: `isinstance(number) is int` used to raise
        # TypeError (isinstance() needs a type argument)
        r = datafile.filter(number=0)
        assert isinstance(r, hpl.Ray)
        assert r.azimuth == 0.0

    def test_scalar_float_azimuth(self, datafile):
        # regression test: only a bare int was wrapped into a list,
        # so a bare float (as azimuth actually is) fell through and
        # broke the later `r.azimuth in azimuth` check
        r = datafile.filter(azimuth=60.0)
        assert isinstance(r, hpl.Ray)
        assert r.azimuth == 60.0

    def test_scalar_float_elevation(self, datafile):
        result = datafile.filter(elevation=85.0)
        assert isinstance(result, list)
        assert len(result) == 6

    def test_list_azimuth(self, datafile):
        result = datafile.filter(azimuth=[0.0, 180.0])
        assert [r.azimuth for r in result] == [0.0, 180.0]

    def test_combined_filters(self, datafile):
        result = datafile.filter(number=[0, 1, 2], azimuth=[0.0, 120.0])
        assert [r.azimuth for r in result] == [0.0, 120.0]

    def test_no_match_returns_empty_list(self, datafile):
        assert datafile.filter(azimuth=999.0) == []


# =========================================================================
# module-level read()
# =========================================================================

class TestRead:

    def test_read_all_variables(self):
        fields = hpl.read(str(WIND_PROFILE_FILE))
        assert set(fields) == {'Doppler', 'Intensity', 'Beta'}
        for df in fields.values():
            assert df.shape == (6, 750)
            assert df.columns[:3].tolist() == [0, 1, 2]

    def test_read_sorted_by_time(self):
        fields = hpl.read(str(WIND_PROFILE_FILE))
        idx = list(fields['Doppler'].index)
        assert idx == sorted(idx)

    def test_read_filtered(self):
        fields = hpl.read(str(WIND_PROFILE_FILE), azimuth=[0.0, 180.0])
        for df in fields.values():
            assert df.shape == (2, 750)

    def test_read_accepts_list_of_patterns(self):
        fields = hpl.read([str(WIND_PROFILE_FILE)])
        assert set(fields) == {'Doppler', 'Intensity', 'Beta'}

    def test_read_no_matching_files_returns_empty_dict(self):
        fields = hpl.read(str(TESTDIR / 'no_such_file_*.hpl'))
        assert fields == {}

    def test_read_ignores_processed_profile_files(self):
        # a "Processed Wind Profile" file has no rays, so it
        # contributes nothing to read()'s result (see its docstring)
        fields = hpl.read(str(PROCESSED_PROFILE_FILE))
        assert fields == {}


# =========================================================================
# header-less "Processed Wind Profile" file:
# Processed_Wind_Profile_77_20260919_121707.hpl
# =========================================================================

class TestProcessedProfileFile:

    @pytest.fixture
    def datafile(self):
        return hpl.DataFile(file=str(PROCESSED_PROFILE_FILE))

    def test_type_and_timestamp_come_from_filename(self, datafile):
        # this file has no header of its own to read them from
        assert datafile.type == 'Processed Wind Profile'
        assert datafile.timestamp == pd.Timestamp('2026-09-19 12:17:07')

    def test_rays_empty_and_vars_none(self, datafile):
        assert datafile.rays == []
        assert datafile.vars is None

    def test_profile_shape_and_index(self, datafile):
        assert datafile.profile.shape == (750, 2)
        assert datafile.profile.index.name == 'height'
        assert datafile.profile.columns.tolist() == ['direction', 'speed']

    def test_profile_dtypes_are_numeric(self, datafile):
        assert all(dt == float for dt in datafile.profile.dtypes)

    def test_profile_first_and_last_row(self, datafile):
        assert datafile.profile.index[0] == pytest.approx(9.0)
        first = datafile.profile.iloc[0]
        assert first['direction'] == pytest.approx(240.00)
        assert first['speed'] == pytest.approx(1.17)

        assert datafile.profile.index[-1] == pytest.approx(2247.4)
        last = datafile.profile.iloc[-1]
        assert last['direction'] == pytest.approx(69.18)
        assert last['speed'] == pytest.approx(90.47)


# =========================================================================
# loading a different file into an existing object
# =========================================================================

def test_load_can_switch_file_variant():
    df = hpl.DataFile(file=str(WIND_PROFILE_FILE))
    assert df.type == 'Wind profile'
    df.load(file=str(PROCESSED_PROFILE_FILE))
    assert df.type == 'Processed Wind Profile'
    assert df.rays == []


# =========================================================================
# regression tests using minimal, synthetic ".hpl" files
# =========================================================================

_MINIMAL_HEADER = """\
Filename:\t{filename}
System ID:\t99
Number of gates:\t{gates}
Range gate length (m):\t30.0
Gate length (pts):\t10
Pulses/ray:\t1
No. of rays in file:\t1
Scan type:\t{scantype}
Focus range:\t65535
Start time:\t{starttime}
Resolution (m/s):\t0.0364
Data line 1: Decimal time (hours)  Azimuth (degrees)  Elevation (degrees) Pitch (degrees) Roll (degrees)
f9.6,1x,f6.2,1x,f6.2
Data line 2: Range Gate  Doppler (m/s)  Intensity (SNR + 1)  Beta (m-1 sr-1)
i3,1x,f6.4,1x,f8.6,1x,e12.6 - repeat for no. gates
****
 0.000000  10.00  45.00   0.00   0.00
  0 0.1000 1.000000  1.000000E-7
  1 0.2000 1.000000  1.000000E-7
"""


def _write_minimal_file(tmp_path, starttime, scantype='User1', gates=2,
                         filename='User1_00_20260101_000000.hpl'):
    """
    Write a minimal, but structurally valid, regular scan file (one
    ray, two gates by default) with the given ``Start time`` and
    ``Scan type`` header values, and return its path.
    """
    path = tmp_path / filename
    path.write_text(_MINIMAL_HEADER.format(
        filename=filename, starttime=starttime, scantype=scantype,
        gates=gates))
    return path


class TestTimestampParsing:

    def test_with_fractional_seconds(self, tmp_path):
        path = _write_minimal_file(tmp_path, '20260101 00:00:00.05')
        df = hpl.DataFile(file=str(path))
        assert df.timestamp == pd.Timestamp('2026-01-01 00:00:00.050000')

    def test_without_fractional_seconds(self, tmp_path):
        # regression test: load() used to crash with
        # "ValueError: time data ... doesn't match format
        # '%Y%m%d %H:%M:%S.%f'" on files whose start time has no
        # fractional part at all
        path = _write_minimal_file(tmp_path, '20260101 00:00:00')
        df = hpl.DataFile(file=str(path))
        assert df.timestamp == pd.Timestamp('2026-01-01 00:00:00')


class TestScanTypeParsing:

    def test_type_without_mode(self, tmp_path):
        path = _write_minimal_file(tmp_path, '20260101 00:00:00',
                                    scantype='User1')
        df = hpl.DataFile(file=str(path))
        assert df.type == 'User1'
        assert df.mode == ''

    def test_type_with_mode(self, tmp_path):
        path = _write_minimal_file(tmp_path, '20260101 00:00:00',
                                    scantype='User1 - stepped')
        df = hpl.DataFile(file=str(path))
        assert df.type == 'User1'
        assert df.mode == 'stepped'

    def test_unknown_scantype_raises(self, tmp_path):
        path = _write_minimal_file(tmp_path, '20260101 00:00:00',
                                    scantype='Bogus')
        with pytest.raises(ValueError, match='unknown scantype'):
            hpl.DataFile(file=str(path))


class TestGateCountValidation:

    def test_mismatched_gate_count_raises(self, tmp_path):
        # header claims 3 gates, but the data block (built with the
        # default gates=2) only has 2 -- _read_ray must catch this
        path = _write_minimal_file(tmp_path, '20260101 00:00:00',
                                    gates=3)
        with pytest.raises(IOError, match='gates read'):
            hpl.DataFile(file=str(path))
