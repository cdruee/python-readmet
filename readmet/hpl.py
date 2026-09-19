#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The classes and functions in this category handle files
in format ".hpl" for wind-lidar raw data
created by Halo phtotonics Ltd., Worcester, United Kingdom.
(https://halo-photonics.com)

The most comprehensive description of ".hpl" can be found at
https://aidaho-edu.uni-hohenheim.de/gitlab/lafe_test/lafi-data-management/-/wikis/home/HPL-Doppler-LiDAR-File-Format-Documentation

"""

import glob
import logging
import re

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------

MARKER = re.compile(r'^\s*[*]{2,}\s*$')


# =========================================================================

class Ray(object):
    """
    Holds one ray (single beam) of profile data.
    """
    time: np.datetime64
    """time of the ray"""
    elevation: float
    """elevation angle in degrees"""
    azimuth: float
    """azimuth angle in degrees"""
    pitch: float
    """pitch angle in degrees"""
    roll: float
    """roll angle in degrees"""
    gates: int
    """number of range gates"""
    data: pd.DataFrame
    """per-gate data: columns gate, doppler, intensity, beta"""

    def __init__(self, time=None, elevation=None, azimuth=None,
                 pitch=None, roll=None, gates=None, df=None):
        self.time = time if time is not None else np.datetime64('NaT')
        self.elevation = elevation
        self.azimuth = azimuth
        self.pitch = pitch
        self.roll = roll
        self.gates = gates
        self.data = df if df is not None else pd.DataFrame(
            columns=['gate', 'doppler', 'intensity', 'beta'])

# =========================================================================

class DataFile(object):
    """
    object class that holds data and metadata of a .hpl file

    :param file: filename (optionally including path). \
      If missing, an emtpy object is returned
    :param text: (optional) If ``True`` the raw file contents \
      are containted as atrribute `text` in the object. If ``False`` \
      or missing, the raw file contents are discarded after parsing.
    """

    file:str | None = None
    """ name of file loaded into object """
    header:dict | None = None
    """ dictionary containing the header entries as strings"""
    vars: pd.DataFrame | None = None
    """ ``pandas.Dataframe`` containing information of the variables
      The index contains the variable symbol.
      The columns are "label","symbol","unit","type","error_mask","gap_value"
      for each variable. """
    rays:list[Ray] | None = None
    """ list of rays in files.
      The index is time, each column represents one variable. """
    profile:pd.DataFrame | None = None
    """ dictonary containing the profile data from the file loaded.
      The keys are the variable names.
      The values are of type ``pandas.DataFrame`` with time as index,
      and the measurement levels as columns. """
    text:list[str] | None = None
    """ text contents the file loaded. Also contains the (decompressed)
      text contents of an eventual external `datfile` appended to
      the main file.
    """
    #
    # evaluate file name
    #
    def _parse_filename(self, name=None):
        if name is None:
            name = self.header['filename']
        if name is None:
            name = self.file
        # remove extension
        name = name.strip('.hpl')
        strings = name.split('_')
        timestamp = pd.to_datetime(''.join(strings[-2:]),
                                   format='%Y%m%d%H%M%S')
        type = strings[:-3]
        return type, timestamp
    #
    # read header "header"
    #
    def _get_header(self):
        #
        # read the file as text lines and check magic
        #
        header = dict()
        text_iter = iter(self.text)
        for line in text_iter:
            # line of stars end the header
            if re.match(MARKER, line):
                break
            try:
                hkey, hval = line.split(':', 1)
            except ValueError:
                continue
            if hkey == 'Filename':
                header['filename'] = hval.strip()
            elif hkey == 'System ID':
                header['instrument'] = hval.strip()
            elif hkey =='Start time':
                header['starttime'] = hval.strip()
            elif hkey == 'Number of gates':
                header['gates'] = hval.strip()
            elif hkey == 'Range gate length (m)':
                header['gatelength'] = hval.strip()
            elif hkey == 'Gate length (pts)':
                header['gatepoints'] = hval.strip()
            elif hkey == 'Pulses/ray':
                header['pulses'] = hval.strip()
            elif hkey == 'No. of rays in file':
                header['rays'] = hval.strip()
            elif hkey == 'Scan type':
                header['scantype'] = hval.strip()
            elif hkey == 'Focus range':
                header['focus'] = hval.strip()
            elif hkey == 'Resolution (m/s)':
                header['resolution'] = hval.strip()
            elif hkey == 'Resolution (m/s)':
                header['resolution'] = hval.strip()
            elif hkey == 'Data line 1' or hkey == 'Data line 2':
                num = int(hkey.split()[2])
                header[f'variables_{num}'] = []
                header[f'units_{num}'] = []
                fields = hval.replace('  ',')').split(')')
                for f in fields:
                    if len(f) > 0:
                        if '(' in f:
                            x,y = f.split('(')
                        else:
                            x = f
                            y = ''
                        header[f'variables_{num}'].append(x.strip())
                        header[f'units_{num}'].append(y.strip())
                next(text_iter)
                header[f'format_{num}'] = line.strip()
            else:
                raise ValueError(
                    f'unknown header {hkey} in {self.file}')
        return header
    #
    # read variable definitions
    #

    def _get_variables(self):
        columns = ["label", "unit", "type"]
        variables = pd.DataFrame(columns=columns)
        variables["label"] = self.header['variables_2']
        variables["unit"] = self.header['units_2']
        # decode types from format string
        formats = [x for x in self.header['format_2'].split(',')
                   if x.endswith('x')]
        types = []
        for f in formats:
            if 'f' in f or 'd' in f or 'g' in f:
                types.append('f')
            elif 'i' in f or 'u' in f:
                types.append('i')
            else:
                raise ValueError(
                    f'unknown format {f} in '
                    f'format {self.header['format_2']}'
                    f'in file {self.file}')
        # some hpl files miss fields in format description
        while len(types) < len(variables["label"]):
            types.append('f')
        variables['type'] = types
        variables.index = variables['label']
        return variables
    #
    # get the data block from file text
    #
    def _get_val(name, names, values):
        try:
            return values[names.index(name)]
        except ValueError:
            return None

    def _read_ray(self,
                  lines: str | list[str],
                  gates: int | None = None,
                  variables_1: list[str] | None = None,
                  variables_2: list[str] | None = None):
        # ensure lines is list of strings
        if isinstance(lines, str):
            lines = lines.split('\n')
        # ensure we have number of gates
        if not gates:
            gates = int(float(self.header['gates']))
        # get attribute values
        if not variables_1:
            variables_1 = self.header['variables_1']
        values_1 = [float(x) for x in lines[0].split()]
        raw_hours = self._get_val('Decimal time', variables_1, values_1)
        hours = pd.Timedelta(
            hours=raw_hours) if raw_hours is not None else None
        azimuth = self._get_val('Azimuth', variables_1, values_1)
        elevation = self._get_val('Elevation', variables_1, values_1)
        pitch = self._get_val('Pitch', variables_1, values_1)
        roll = self._get_val('Roll', variables_1, values_1)
        # construct ray time by replacing the hours in the file timestamp
        if hours is None:
            time = self.timestamp
        else:
            daystart = self.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            time = daystart + hours
        # get the values along the ray
        df = pd.DataFrame([line.split() for line in lines[1:]])
        if len(df.index) != gates:
            raise IOError(f'gates read: {len(df.index)} '
                          f'but expected {gates}')
        if variables_2:
            if len(df.columns) != len(variables_2):
                raise ValueError(
                    f"variables_2 has {len(variables_2)} "
                    f"elements but data have {len(df.columns)} columns")
            df.columns = variables_2
        #
        # put it into a ray object
        return Ray(time=time, elevation=elevation, azimuth=azimuth,
                   pitch=pitch, roll=roll, gates=gates, df=df)
    #
    # read the datablock into rays
    #
    def _get_datablock(self):
        # fast forward to data:
        for nl, line in enumerate(self.text):
            # line of stars end the header
            if re.match(MARKER, line):
                data_start = nl + 1
                break
        else:
            raise IOError('data block not found in file')
        nrays = int(float(self.header['rays']))
        ngates = int(float(self.header['gates']))
        rays = []
        for n in range(nrays):
            ray_start = data_start + n * (ngates + 1)
            ray_end = data_start + (n + 1) * (ngates + 1)
            ray = self._read_ray(self.text[ray_start:ray_end])
            rays.append(ray)
        return rays
    #
    # read profile data
    #
    def _get_profile(self):
        return None
    #
    # read file into memory
    #

    def load(self, file=None, text=False):
        if file is not None:
            self.file = file
        with open(self.file, 'r') as f:
            self.text = [x.rstrip() for x in f.readlines()]
        self.header = self._get_header()
        self.type,self.timestamp = self._parse_filename()
        self.vars = self._get_variables()
        self.rays = self._get_datablock()
        self.profile = self._get_profile()
        if text is not True:
            del self.text
    #
    # constructor
    #

    def __init__(self, file=None, text=None):
        object.__init__(self)
        self.file = file
        if file is not None:
            self.load(file, text)

# ------------------------------------------------------------------------


# def read(pattern):
#     """
#     read a sequence of Scintec-1 files into one data structue
#
#     :param pattern: a `globbing pattern \
#         <https://en.wikipedia.org/wiki/Glob_(programming)>`_ \
#         describing one or multiple filenames or paths
#     :returns: contained variables as dictionary with variable names as keys. \
#           Each variable is returned as a `pandas.DataFrame \
#           <https://pandas.pydata.org/pandas-docs/stable\
# /reference/api/pandas.DataFrame.html>`_ \
#       with date/time as index of type `pandas.DatetimeIndex \
#           <https://pandas.pydata.org/pandas-docs/stable\
# /reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex>`_
#     """
#     # expand globbing pattern
#     if isinstance(pattern, list):
#         files = []
#         for x in pattern:
#             # expand globbing pattern
#             logging.debug('globbing pattern is: {}'.format(x))
#             ex = glob.glob(x)
#             logging.debug('expanded file list : {}'.format(ex))
#             # append to full list of files:
#             files += ex
#     else:
#         files = glob.glob(pattern)
#     logging.debug('list of files to open: {}'.format(files))
#     # read all files
#     fields = None
#     series = None
#     for i, file in enumerate(files):
#         logging.info('opening file #{}:{}'.format(i, file))
#         scintec1 = DataFile(file)
#         if len(scintec1.vars) > 0:
#             if fields is None and series is None:
#                 fields = scintec1.profile
#                 series = scintec1.nonprofile
#             else:
#                 warn_duplicated = False
#
#                 # append profile variables, if any
#                 fmore = scintec1.profile
#                 # only in case there ara data to append (to)
#                 if fields is not None and fmore is not None:
#                     # go through all the variables
#                     for c in fmore.keys():
#                         # look if we have these variable in stock
#                         if c in fields.keys():
#                             # check that types match
#                             if isinstance(fmore[c], type(fields[c])):
#                                 # issue a warning if we have duplicate times
#                                 if any(x in fields[c].index
#                                        for x in fmore[c].index):
#                                     warn_duplicated = True
#                                 # append data
#                                 # .drop_duplicates(keep='last')
#                                 fields[c] = pd.concat([fields[c], fmore[c]])
#                             else:
#                                 raise TypeError('dont know how to handle ' +
#                                                 ' variable {}'.format(c))
#                         else:
#                             logging.warning('new variable ' +
#                                          '"{}" in file {}'.format(
#                                              c, scintec1.file))
#                             logging.warning('{}'.format(fmore.keys()))
#
#                 # append non-profile variables, if any
#                 smore = scintec1.nonprofile
#                 if series is not None and smore is not None:
#                     if any(x in series.index for x in smore.index):
#                         warn_duplicated = True
#                     series = pd.concat(
#                         [series, smore]).drop_duplicates(keep='last')
#
#                 if warn_duplicated is True:
#                     logging.warning('repeated times in file {}')
#         del scintec1
#     # sort profile data by time
#     if fields is not None and len(fields) > 0:
#         for c in fields.keys():
#             if isinstance(fields[c], pd.DataFrame):
#                 fields[c].sort_index(inplace=True)
#             else:
#                 raise ValueError(
#                     'sort time: dont know how to handle field %s' % str(c))
#     # sort non-profile data by time
#     if series is not None and len(series.keys()) > 0:
#         if isinstance(series, pd.DataFrame):
#             series.sort_index(inplace=True)
#         else:
#             raise ValueError(
#                 'sort time: dont know how to handle series %s' % str(series))
#         # add to profile data
#         for c in series.keys():
#             if not (c == "time" and "time" in fields.keys()):
#                 fields[c] = pd.DataFrame(series[c])
#
#     return fields
