#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The classes and functions in this category handle files
in format "akterm" by German weather service (DWD)

The original columns in dat file specification are:

+-------+--------------------------------------+-------------+
| entry | description                          | data range  |
+-------+--------------------------------------+-------------+
| KENN  | Kennung für das Datenkollektiv       | ``AK``      |
+=======+======================================+=============+
| STA   | Stationsnummer                       | 00001-99999 |
+-------+--------------------------------------+-------------+
| JAHR  | Jahr                                 | 1800-2999   |
+-------+--------------------------------------+-------------+
| MON   | Monat                                | 1-12        |
+-------+--------------------------------------+-------------+
| TAG   | Tag                                  | 1-31        |
+-------+--------------------------------------+-------------+
| STUN  | Stunde                               | 0-23        |
+-------+--------------------------------------+-------------+
| NULL  | --                                   | 0           |
+-------+--------------------------------------+-------------+
| QDD   | Qualitätsbyte (Windrichtung)         | 0,1,2,9     |
+-------+--------------------------------------+-------------+
| QFF   | Qualitätsbyte (Windgeschwindigkeit)  | 0,1,2,3,9   |
+-------+--------------------------------------+-------------+
| DD    | Windrichtung                         | 0-360,999   |
+-------+--------------------------------------+-------------+
| FF    | Windgeschwindigkeit                  | 0-999       |
+-------+--------------------------------------+-------------+
| QQ1   | Qualitätsbyte (Wertstatus)           | 0-5,9       |
+-------+--------------------------------------+-------------+
| KM    | Ausbreitungsklasse nach Klug/Manier  | 1-7,9       |
+-------+--------------------------------------+-------------+
| QQ2   | Qualitätsbyte (Wertstatus)           | 0,1,9       |
+-------+--------------------------------------+-------------+
| HM    | Mischungsschichthöhe (m)             | 0-9999      |
+-------+--------------------------------------+-------------+
| QQ3   | Qualitätsbyte (Wertstatus)           | 0-5,9       |
+-------+--------------------------------------+-------------+
| PP    | Niederschlag (SYNOP Code)            | 0-999       |
| QPP   | Qualitätsbyte Niederschlag           | 0,9         |
+-------+--------------------------------------+-------------+

"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
#
#
_AKT_COLUMNS = ['KENN', 'STA', 'JAHR', 'MON', 'TAG', 'STUN', 'NULL',
                'QDD', 'QFF', 'DD', 'FF', 'QQ1', 'KM', 'QQ2', 'HM', 'QQ3']
_AKTN_COLUMNS = _AKT_COLUMNS + ['PP', 'QPP']
_PREC_KEYWORD = 'Niederschlag'
_displacement_factor = 6.5
#
z0_classes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 1.5, 2]
#
# ------------------------------------------------------------------------
#
def precipitation_to_synop(precip):
    """
    generate SYNOP code numbers for precipitation amount

    Parameters
    ----------
    precip : pandas.Series or array-like
        precipitation amount in mm / observation period

    Returns
    -------
    s : pandas.Series
        SYNOP key PP (precipitation amount):
            000    no precipitation
            001    1 mm
            002    2 mm
            ...    ...
            988    988 mm
            989    989 mm or more
            990    0.05 mm or less
            991    0.1 mm
            992    0.2 mm
            ...    ...
            999    0.9 mm
    """
    p = pd.Series(precip).round(1)
    s = pd.Series(np.nan, index=p.index)
    for k,v in p.items():
        if np.isnan(v) or v < 0.:
           i = np.nan
        elif v == 0.:
            i = 0
        elif v < 1.:
            i = 990 + int(10 * v)
        elif v <= 988.:
            i = int(v)
        elif v > 988.:
            i = 989
        else:
            raise ValueError('internal error for rain amount: %f' % format(v))
        s.loc[k] = i
    return s

def precipitation_from_synop(synop):
    """
    calculate precipitation amount from SYNOP code numbers

    Parameters
    ----------
    synop : pandas.Series or array-like
        SYNOP code numbers

    Returns
    -------
    p : pandas.Series
        precipitation amount in mm
    """
    s = pd.Series(synop).astype(float)
    p = pd.Series(np.nan, index=s.index)
    for k,v in s.items():
        if np.isnan(v) or v < 0.:
           i = np.nan
        elif v == 0.:
            i = 0.
        elif v < 990.:
            i = v
        elif v < 991.:
            i = 0.5
        elif v <= 999.:
            i = (v - 990.) / 10.
        else:
            raise ValueError('internal error synop code: %.0f' % format(v))
        p.loc[k] = i
    return p

class DataFile(object):
    """
    object class that holds data and metadata of a dmna file

    :param file: (optional, string) filename (optionally including path). \
        If missing, an emtpy object is returned
    :param data: (optional, padas.DataFrame) timeseries data. \
        Expected format: Time (datetime64) as data index,
        wind speed in m/s in column ``FF``, winddirection din degrees in
        column ``DD``, stability class in column ``KM``.
        Precipitation amount in mm column `PP` if prec is True.
    :param z0: (optional, float) surface roughness lenght in m.
        This parameter is only used if `data` is given.
        If ``None`` or missing, effective anemometer height are set to 0.
    :param has: (optional, float) height of the anemometer in m.
        This parameter is only used if `data` is given.
        If missing, 10 m is used.
    :param prec: (optional, bool or None) If True file / data
        contains additional column `PP` for rain amount in mm.
        Default to None.
    """

    file = None
    """ name of file loaded into object """
    header = None
    """ array containing the header lines as strings"""
    vars = None
    """ Number of variables in file    """
    heights = None
    """ effective anemometer heights for each rouchness class in m"""
    data = None
    """ DataFrame containing the data from the file loaded.
    The orginal columns KENN, 'JAHR', 'MON', 'TAG', 'STUN', 'NULL'
    are not contained in the DataFrame, instead the date and time
    are given in the index (datetime64).
    """
    prec = False
    """ file is extended AKTerm Format containing additional columns
    containing precipitation infromation
    """
    # ----------------------------------------------------------------------
    #
    # read header
    #

    def _get_header(self, f=None):
        """
        parses the file as text, finds the divider line "*"
        and returns the header as dictionary
        """
        if f is None:
            f = self.file
        header = []
        f.seek(0)
        for line in f:
            stripped = line.strip()
            if stripped.startswith("*"):
                header.append(stripped[1:].strip())
                logger.debug('header: %s' % header[-1])
            else:
                break
        return header
    # ----------------------------------------------------------------------
    #
    # read header
    #

    def _get_heights(self, f=None):
        """
        parses the file as text, line prefixed by "+"
        and returns the effective anemometer heights
        """
        if f is None:
            f = self.file
        heights = []
        f.seek(0)
        for line in f.readlines():
            stripped = line.strip()
            if stripped.startswith("+"):
                numstr = stripped.split(':')[1]
                heights = np.fromstring(numstr, dtype=int, sep=' ') * 0.1
                logger.debug('heights: %s' % format(heights))
                break
        return heights

    # ----------------------------------------------------------------------
    #
    # read data
    #
    def _get_data(self, f=None, prec=False):
        """
        parses the file as text, skips header
        and returns the data as dataframe
        if prec is True, precipitation columns are read
        """
        if f is None:
            f = self.file
        header_lines = 0
        f.seek(0)
        for line in f.readlines():
            header_lines = header_lines + 1
            if line.lstrip().startswith("+"):
                break
        f.seek(0)
        if prec:
            akt_columns = _AKTN_COLUMNS
        else:
            akt_columns = _AKT_COLUMNS

        data = pd.read_csv(f,
                           sep='\\s+',
                           skiprows=header_lines,
                           engine='python',
                           names=akt_columns)
        #
        # apply quality flags
        #
        # 0 Windgeschwindigkeit in Knoten
        data['FF'] = data['FF'].mask(
            data['QFF'] == 0, data['FF'] * 0.514, axis=0)
        # 1 Windgeschwindigkeit in 0,1 m/s, Original in 0,1 m/s
        data['FF'] = data['FF'].mask(
            data['QFF'] == 1, data['FF'] * 0.1, axis=0)
        # 2 Windgeschwindigkeit in 0,1 m/s, Original in Knoten (0,514 m/s)
        data['FF'] = data['FF'].mask(
            data['QFF'] == 2, data['FF'] * 0.1, axis=0)
        # 3 Windgeschwindigkeit in 0,1 m/s, Original in m/s
        data['FF'] = data['FF'].mask(
            data['QFF'] == 3, data['FF'] * 0.1, axis=0)
        # 9 Windgeschwindigkeit fehlt
        data['FF'] = data['FF'].mask(data['QFF'] == 9, np.nan, axis=0)
        #
        # 0 Windrichtung in Dekagrad
        data['DD'] = data['DD'].mask(data['QDD'] == 0, data['DD'] * 10, axis=0)
        # 1 Windrichtung in Grad, Original in Dekagrad
        data['DD'] = data['DD'].mask(data['QDD'] == 1, data['DD'], axis=0)
        # 2 Windrichtung in Grad, Original in Grad
        data['DD'] = data['DD'].mask(data['QDD'] == 2, data['DD'], axis=0)
        # 9 Windrichtung fehlt
        data['DD'] = data['DD'].mask(data['QDD'] == 9, np.nan, axis=0)
        #
        # 9 Niederschlag fehlt oder verdächtig
        if prec:
            # 9 Niederschlag fehlt; SYNOP code -> mm
            data['PP'] = precipitation_from_synop(
                data['PP'].mask(data['QPP'] == 9, np.nan, axis=0)
            )

        #
        # Make datetime:
        data.index = pd.to_datetime(pd.DataFrame({'year': data['JAHR'],
                                     'month': data['MON'],
                                     'day': data['TAG'],
                                     'hour': data['STUN']}))
        data.drop(columns=['KENN', 'JAHR', 'MON', 'TAG', 'STUN', 'NULL'])
        return data
    # ----------------------------------------------------------------------
    #
    # output data
    #

    def _out_data(self, prec=False):
        """
        prepare DataFrame consistent and in proper unis for output
        """
        out = self.data.copy()
        if 'KENN' not in out.columns:
            out['KENN'] = 'AK'
        if 'STA' not in out.columns:
            out['STA'] = 10999
        for x in ['FF', 'DD']:
            q = 'Q' + x
            if q not in out.columns:
                out[q] = 0
            # flag 9 marks "no value"
            out[q] = out[q].mask(out[x].isna(), 9)
        # value 7 marks "no value"
        out['KM'] = out['KM'].mask(out['KM'].isna(), 7)
        for q in ['QQ1', 'QQ2', 'QQ3']:
            if q not in out.columns:
                out[q] = 0
        if prec:
            if 'QPP' not in out.columns:
                out['QPP'] = 1
        if 'HM' not in out.columns:
            out['HM'] = -9999.
            out['QQ3'] = 9
        #
        # split datetime into columns
        #
        out['JAHR'] = out.index.year
        out['MON'] = out.index.month
        out['TAG'] = out.index.day
        out['STUN'] = out.index.hour
        out['NULL'] = 0
        #
        # apply quality flags
        #
        # 0 Windgeschwindigkeit in Knoten
        out['FF'] = out['FF'].mask(out['QFF'] == 0, out['FF'] / 0.514, axis=0)
        # 1 Windgeschwindigkeit in 0,1 m/s, Original in 0,1 m/s
        out['FF'] = out['FF'].mask(out['QFF'] == 1, out['FF'] / 0.1, axis=0)
        # 2 Windgeschwindigkeit in 0,1 m/s, Original in Knoten (0,514 m/s)
        out['FF'] = out['FF'].mask(out['QFF'] == 2, out['FF'] / 0.1, axis=0)
        # 3 Windgeschwindigkeit in 0,1 m/s, Original in m/s
        out['FF'] = out['FF'].mask(out['QFF'] == 3, out['FF'] / 0.1, axis=0)
        # 9 Windgeschwindigkeit fehlt
        out['FF'] = out['FF'].mask(out['QFF'] == 9, 99, axis=0)
        #
        # 0 Windrichtung in Dekagrad
        out['DD'] = out['DD'].mask(out['QDD'] == 0, out['DD'] / 10, axis=0)
        # 1 Windrichtung in Grad, Original in Dekagrad
        out['DD'] = out['DD'].mask(out['QDD'] == 1, out['DD'], axis=0)
        # 2 Windrichtung in Grad, Original in Grad
        out['DD'] = out['DD'].mask(out['QDD'] == 2, out['DD'], axis=0)
        # 9 Windrichtung fehlt
        out['DD'] = out['DD'].mask(out['QDD'] == 9, 999, axis=0)
        #
        if prec:
            # 9 Niederschlag fehlt; mm -> SYNOP code
            out['PP'] = precipitation_to_synop(
                out['PP'].mask(out['QPP'] == 9, np.nan, axis=0)
            )
        # make columns integer
        if prec:
            akt_columns = _AKTN_COLUMNS
        else:
            akt_columns = _AKT_COLUMNS
        for c in akt_columns:
            if c != 'KENN':
                try:
                    out[c] = out[c].map(np.round).map(int)
                except Exception as e:
                    logger.error('column did not convert: ' + c)
                    raise e
        #
        # reorder columns:
        out = out[[*akt_columns]]
        #
        # write into string
        res = out.to_string(index=False,
                            header=False,
                            formatters={'KENN': '{:2s}'.format,
                                        'STA': '{:5d}'.format,
                                        'JAHR': '{:4d}'.format,
                                        'MON': '{:02d}'.format,
                                        'TAG': '{:02d}'.format,
                                        'STUN': '{:02d}'.format,
                                        'NULL': '{:02d}'.format,
                                        'QFF': '{:1d}'.format,
                                        'QDD': '{:1d}'.format,
                                        'DD': '{:3d}'.format,
                                        'FF': '{:3d}'.format,
                                        'QQ1': '{:1d}'.format,
                                        'KM': '{:1d}'.format,
                                        'QQ2': '{:1d}'.format,
                                        'HM': '{:4d}'.format,
                                        'QQ3': '{:1d}'.format,
                                        'PP': '{:3d}'.format,
                                        'QPP': '{:1d}'.format, }
                            )
        # AK 10999 1995 01 01 00 00 1 1 210 56 1 3 1 -999 9

        return res
    #
    # ----------------------------------------------------------------------
    #
    # get effective anemometer height from object/file
    #

    def get_h_anemo(self, z0=None):
        """
        returns the effective anemometer height(s) from the object

        :param z0: roughness length for which the effective anemometer height
            should be determined.
            If missing, all heights are returned as array
        :return: effective anemometer height in m
        :rtype: float or array
        """
        if z0 is None:
            re = self.heights
        else:
            for he, zc in zip(self.heights, z0_classes):
                if z0 == zc:
                    re = he
                    break
            else:
                raise ValueError('not a z0 class: %f' % z0)
        return re
    #
    # ----------------------------------------------------------------------
    #
    # set effective anemometer heights
    #

    def set_h_anemo(self, z0=None, has=None):
        """
        sets the effective anemometer height(s) from z0 and has
        :param z0: roughness length at the site of the wind measurement in m
        :param has: height of the wind measurement in m.
            If ``None`` or missing, 10 m is used.
        """
        if has is None:
            has = 10.
        self.heights = h_eff(z0, has)
        return
    #
    # ----------------------------------------------------------------------
    #
    # read data
    #

    def write(self, file=None):
        if file is None:
            path = self.file
        else:
            path = file
        with open(path, 'w') as f:
            #
            # header
            for line in self.header:
                f.write('* {:<78s}\r\n'.format(line))
            # anemometer heights
            height_line = '+ Anemometerhoehen (0.1 m):'
            for h in self.heights:
                height_line += ' {:4d}'.format(int(h * 10))
            f.write(height_line + '\r\n')
            # data
            block = self._out_data(self.prec)
            for line in block.splitlines():
                f.write(line.lstrip() + '\r\n')
    #
    # ----------------------------------------------------------------------
    #
    # read file into memory
    #

    def load(self, file):
        """
        loads the contents of an akterm file into the object

        :param file: filename (optionally including path). \
            If missing, an emtpy
        :return: DataFrame with datetime as index, FF in m/s,
            DD in DEG, an KM Korman/Meixner stability class,
            columns QDD,QFF,QQ1,QQ1,HM are contained as in the
            file.
        """
        logger.info('loading file: %s' % file)
        with open(self.file, 'r') as f:
            self.header = self._get_header(f)
            if len(self.header) >= 1 and _PREC_KEYWORD in self.header[0]:
                self.prec = True
            else:
                self.prec = False
            self.heights = self._get_heights(f)
            self.data = self._get_data(f, self.prec)
    #
    # ----------------------------------------------------------------------
    #
    # constructor
    #

    def __init__(self, file=None, data=None, z0=None, has=None, prec=None):
        object.__init__(self)
        self.file = file
        if file is not None:
            if data is not None or z0 is not None:
                raise ValueError('data and z0 must be None if file is given')
            elif prec is not None:
                raise ValueError('prec must not be given if file is given')
            else:
                self.load(file)
        else:
            if isinstance(data, dict):
                self.data = pd.DataFrame.from_dict(data)
            else:
                self.data = pd.DataFrame(data)
            if z0 is None:
                self.heights = [0] * len(z0_classes)
            else:
                self.set_h_anemo(z0, has)
            self.file = None
            if prec not in [None, True, False]:
                raise ValueError('prec must be either boolean or None')
            if prec is None or prec is False:
                self.header = []
                self.prec = False
            elif prec:
                self.header = [_PREC_KEYWORD, ]
                self.prec = True

# ----------------------------------------------------


def h_eff(z0s, has):
    """
    calulate effectice anemometer heights for all
    roughness-length classes used in asutal2000
    :param z0s: roughness length at the site of the wind measurement in m
    :param has: height of the wind measurement in m
    :return: list of length 9 containing the nine
        effecive anemometer heights
    :rtype: list(9)
    """
    href = 250
    d0s = _displacement_factor * z0s
    ps = np.log((has - d0s) / z0s) / np.log((href - d0s) / z0s)
    ha = []
    for z0 in z0_classes:
        d0 = _displacement_factor * z0
        ha.append(d0 + z0 * ((href - d0) / z0)**ps)
    return ha

# ----------------------------------------------------


if __name__ == '__main__':
    #    import matplotlib.pyplot as plt
    #    from matplotlib import cm
    logger.setLevel(logging.DEBUG)
    #
    # test axes
    qq = DataFile('../tests/anno95.akterm')
    print(qq.data[['KENN', 'FF', 'DD']])
    qq.write('../tests/out.akterm')
