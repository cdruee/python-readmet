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
TYPES = ['VAD', 'RHI', 'Wind profile', 'User1',
         'User2', 'User3', 'User4', 'User5',
         'Processed Wind Profile']


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
    timstamp: pd.Timestamp | None = None
    type:str | None = None
    f""" scan type string, one of {TYPES}     
    """
    overlapping:bool = False
    """ Flag if the scan pattern uses overlapping gates
    """
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
                # NOTE: next(text_iter) on its own only *advances* the
                # iterator, it doesn't rebind `line`.
                fmt_line = next(text_iter)
                header[f'format_{num}'] = fmt_line.strip()
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
        # decode types from format string, e.g.
        # "i3,1x,f6.4,1x,f8.6,1x,e12.6 - repeat for no. gates"
        # "1x" tokens are Fortran-style single-character spacers,
        # keep everything *except* those.
        formats = [x for x in self.header['format_2'].split(',')
                   if not x.endswith('x')]
        types = []
        for f in formats:
            # the last format code can carry a trailing free-text
            # comment (as above); strip it before classifying, or
            # letters from the comment itself can be mistaken for
            # format-code letters.
            code = f.split(' - ')[0].strip()
            if any(c in code for c in ('f', 'd', 'g', 'e')):
                types.append('f')
            elif any(c in code for c in ('i', 'u')):
                types.append('i')
            else:
                fmt_2 = self.header["format_2"]
                raise ValueError(
                    f'unknown format {code} in '
                    f'format {fmt_2} '
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
    def _get_val(self, name, names, values):
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
            ray = self._read_ray(self.text[ray_start:ray_end],
                                  gates=ngates,
                                  variables_1=self.header['variables_1'],
                                  variables_2=self.header['variables_2'])
            rays.append(ray)
        return rays
    #
    # read profile data
    #
    def _get_profile(self):
        return None
    #
    #
    #
    def _parse_scantype(self, scantype):
        if '-' in scantype:
            typestring, overstring = scantype.split('-')
        else:
            typestring = scantype
            overstring = ''
        if overstring.strip() == 'overlapping':
            overlapping = True
        else:
            overlapping = False
        typestring = typestring.strip()
        if typestring.upper() not in [x.upper() for x in TYPES]:
            raise ValueError(f'unknown scantype {typestring}')
        return typestring, overlapping

    #
    # read file into memory
    #

    def load(self, file=None, text=False):
        if file is not None:
            self.file = file
        with open(self.file, 'r') as f:
            self.text = [x.rstrip() for x in f.readlines()]
        self.header = self._get_header()
        self.timestamp = pd.to_datetime(self.header['starttime'].strip(),
                                        format='%Y%m%d %H:%M:%S')
        self.type, self.overlapping = self._parse_scantype(
            self.header['scantype'])
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

    def filter(self, number: int | list[int]=None,
               azimuth: float | list[float] | None=None,
               elevation:float | list[float] | None=None):
        out = []
        if number is not None:
            if isinstance(number) is int:
                number = [number]
        if azimuth is not None:
            if type(azimuth) is int:
                azimuth = [azimuth]
        if elevation is not None:
            if type(elevation) is int:
                elevation = [elevation]
        for i,r in enumerate(self.rays):
            if (((number is None) or (i in number)) and
                ((azimuth is None) or (r.azimuth in azimuth)) and
                ((elevation is None) or (r.elevation in elevation))
            ):
                out.append(r)
        if len(out) == 1:
            return out[0]
        else:
            return out

# ------------------------------------------------------------------------
#
# evaluate file name
#
def parse_filename(self, name=None):
    if name is None:
        name = self.header['filename']
    if name is None:
        name = self.file
    # remove extension
    name = name.removesuffix('.hpl')
    strings = name.split('_')
    timestamp = pd.to_datetime(''.join(strings[-2:]),
                               format='%Y%m%d%H%M%S')
    type = strings[:-3]
    return type, timestamp
