#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The classes and functions in this category handle files
in format "akterm" by German weather service (DWD)

The original columns in dat file specification are:
+-------+--------------------------------------+-------------+
| entry | description                          | data range  |
+-------+--------------------------------------+-------------+
| KENN  | Kennung für das Datenkollektiv       | ``AK``      |
| STA   | Stationsnummer                       | 00001-99999 |
| JAHR  | Jahr                                 | 1800-2999   |
| MON   | Monat                                | 1-12        |
| TAG   | Tag                                  | 1-31        |
| STUN  | Stunde                               | 0-23        |
| NULL  | --                                   | 0           |
| QDD   | Qualitätsbyte (Windrichtung)         | 0,1,2,9     |
| QFF   | Qualitätsbyte (Windgeschwindigkeit)  | 0,1,2,3,9   |
| DD    | Windrichtung                         | 0-360,999   |
| FF    | Windgeschwindigkeit                  | 0-999       |
| QQ1   | Qualitätsbyte (Wertstatus)           | 0-5,9       |
| KM    | Ausbreitungsklasse nach Klug/Manier  | 1-7,9       |
| QQ2   | Qualitätsbyte (Wertstatus)           | 0,1,9       |
| HM    | Mischungsschichthöhe (m)             | 0-9999      |
| QQ3   | Qualitätsbyte (Wertstatus)           | 0-5,9       |
+-------+--------------------------------------+-------------+
'''

import logging
import numpy as np
import pandas as pd

from .wind import displacement_factor
#
#
_AKT_COLUMNS=['KENN', 'STA', 'JAHR','MON', 'TAG', 'STUN', 'NULL',
              'QDD', 'QFF','DD', 'FF', 'QQ1', 'KM', 'QQ2', 'HM','QQ3',]
#
#
# ------------------------------------------------------------------------
#

class DataFile(object):
    '''
    object class that holds data and metadata of a dmna file

    :param file: filename (optionally including path). \
        If missing, an emtpy object is returned
    :param text: (optional) If ``True`` the raw file contents \
        are containted as atrribute `text` in the object. If ``False`` \
        or missing, the raw file contents are discarded after parsing.
    '''

    file = None
    ''' name of file loaded into object '''
    header = None
    ''' array containing the header lines as strings'''
    vars = None
    ''' Number of variables in file    '''
    heights = None
    ''' effective anemometer heights for each rouchness class in m'''
    data = None
    ''' DataFrame containing the data from the file loaded.
    The orginal columns KENN, 'JAHR', 'MON', 'TAG', 'STUN', 'NULL'
    are not contained in the DataFrame, instead the date and time
    are given in the index (datetime64).
    '''
    _DF_COLUMNS=[
            'STA', 'QDD', 'QFF','DD', 'FF', 'QQ1', 'KM', 'QQ2', 'HM','QQ3',]
    # ----------------------------------------------------------------------
    #
    # read header
    #
    def _get_header(self, f):
        '''
        parses the file as text, finds the divider line "*"
        and returns the header as dictionary
        '''
        header=[]
        f.seek(0)
        for line in f:
            l = line.strip()
            if l.startswith("*"):
                header.append(l[1:])
                logging.debug('header: %s'%header[-1])
            else:
                break
        return(header)
    # ----------------------------------------------------------------------
    #
    # read header
    #
    def _get_heights(self, f):
        '''
        parses the file as text, line prefixed by "+"
        and returns the header as dictionary
        '''
        heights=[]
        f.seek(0)
        for line in f.readlines():
            l = line.strip()
            if l.startswith("+"):
                numstr = l.split(':')[1]
                heights = np.fromstring(numstr, dtype=int, sep=' ') * 0.1
                logging.debug('heights: %s'%format(heights))
                break
        return(heights)

    # ----------------------------------------------------------------------
    #
    # read data
    #
    def _get_data(self, f):
        '''
        parses the file as text, skips header
        and returns the data as dataframe
        '''
        header_lines=0
        f.seek(0)
        for line in f.readlines():
            header_lines = header_lines + 1
            if line.lstrip().startswith("+"):
                break
        f.seek(0)
        data = pd.read_csv(f, sep='\s+', skiprows=header_lines, engine='python',
                           names=_AKT_COLUMNS)
        #
        # apply quality flags
        #
        # 0 Windgeschwindigkeit in Knoten
        data['FF'] = data['FF'].mask(data['QFF'] == 0, data['FF'] * 0.514, axis=0)
        # 1 Windgeschwindigkeit in 0,1 m/s, Original in 0,1 m/s
        data['FF'] = data['FF'].mask(data['QFF'] == 1, data['FF'] * 0.1, axis=0)
        # 2 Windgeschwindigkeit in 0,1 m/s, Original in Knoten (0,514 m/s)
        data['FF'] = data['FF'].mask(data['QFF'] == 2, data['FF'] * 0.1, axis=0)
        # 3 Windgeschwindigkeit in 0,1 m/s, Original in m/s
        data['FF'] = data['FF'].mask(data['QFF'] == 3, data['FF'] * 0.1, axis=0)
        # 9 Windgeschwindigkeit fehlt
        data['FF'] = data['FF'].mask(data['QFF'] == 9, np.nan,          axis=0)
        #
        # 0 Windrichtung in Dekagrad
        data['DD'] = data['DD'].mask(data['QDD'] == 0, data['DD'] * 10, axis=0)
        # 1 Windrichtung in Grad, Original in Dekagrad
        data['DD'] = data['DD'].mask(data['QDD'] == 1, data['DD'],      axis=0)
        # 2 Windrichtung in Grad, Original in Grad
        data['DD'] = data['DD'].mask(data['QDD'] == 2, data['DD'],      axis=0)
        # 9 Windrichtung fehlt
        data['DD'] = data['DD'].mask(data['QDD'] == 9, np.nan,          axis=0)
        #
        # Make datetime:
        data.index = pd.to_datetime({'year': data['JAHR'],
                                     'month': data['MON'],
                                     'day': data['TAG'],
                                     'hour': data['STUN']})
        data.drop(columns=['KENN', 'JAHR','MON', 'TAG', 'STUN', 'NULL'])
        return(data)
    # ----------------------------------------------------------------------
    #
    # ouput data
    #
    def _out_data(self):
        '''
        prepare DataFrame consistent and in proper unis for output
        '''
        out = self.data.copy()
        if 'KENN' not in out.columns:
                out['KENN'] = 'AK'
        if 'STA' not in out.columns:
                out['STA'] = 10999
        for x in ['FF', 'DD']:
            q = 'Q'+x
            if q not in out.columns:
                out[q] = 0
            out[q].mask(out[x].isna(), 9, inplace=True)
        for q in ['QQ1', 'QQ2', 'QQ3']:
            if q not in out.columns:
                out[q] = 0
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
        out['FF'] = out['FF'].mask(out['QFF'] == 9, 99,              axis=0)
        #
        # 0 Windrichtung in Dekagrad
        out['DD'] = out['DD'].mask(out['QDD'] == 0, out['DD'] / 10, axis=0)
        # 1 Windrichtung in Grad, Original in Dekagrad
        out['DD'] = out['DD'].mask(out['QDD'] == 1, out['DD'],      axis=0)
        # 2 Windrichtung in Grad, Original in Grad
        out['DD'] = out['DD'].mask(out['QDD'] == 2, out['DD'],      axis=0)
        # 9 Windrichtung fehlt
        out['DD'] = out['DD'].mask(out['QDD'] == 9, 999,            axis=0)
        #
        # make columns integer
        for c in _AKT_COLUMNS:
            if c != 'KENN':
                out[c] = out[c].map(np.round).map(int)
        #
        # reorder columns:
        out = out[[*_AKT_COLUMNS]]
        #
        # write into string
        res = out.to_string(index=False,
                 header=False,
                 formatters={'KENN': '{:2s}'.format,
                             'STA':  '{:5d}'.format,
                             'JAHR': '{:4d}'.format,
                             'MON': '{:02d}'.format,
                             'TAG': '{:02d}'.format,
                             'STUN':'{:02d}'.format,
                             'NULL':'{:02d}'.format,
                             'QFF':  '{:1d}'.format,
                             'QDD':  '{:1d}'.format,
                             'DD':   '{:3d}'.format,
                             'FF':   '{:3d}'.format,
                             'QQ1':  '{:1d}'.format,
                             'KM':   '{:1d}'.format,
                             'QQ2':  '{:1d}'.format,
                             'HM':   '{:4d}'.format,
                             'QQ3':  '{:1d}'.format,}
                 )
            #AK 10999 1995 01 01 00 00 1 1 210 56 1 3 1 -999 9

        return res

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
            for l in self.header:
                f.write('* {}\n'.format(l))
            # anemometer heights
            height_line = '+ Anemometerhoehen (0.1 m):'
            for h in self.heights:
                height_line += ' {:4d}'.format(int(h*10))
            f.write(height_line+'\n')
            # data
            block = self._out_data()
            for l in block.splitlines():
                f.write(l.lstrip()+'\n')


    # ----------------------------------------------------------------------
    #
    # read file into memory
    #
    def load(self,file):
        '''
        loads the contents of a akterm file into the object

        :param file: filename (optionally including path). \
            If missing, an emtpy
        :return: DataFrame with datetime as index, FF in m/s,
            DD in DEG, an KM Korman/Meixner stability class,
            columns QDD,QFF,QQ1,QQ1,HM are contained as in the
            file.
        '''
        with open(self.file,'r') as f:
            self.header = self._get_header(f)
            self.heights = self._get_heights(f)
            self.data   = self._get_data(f)
    #
    # ----------------------------------------------------------------------
    #
    # constructor
    #
    def __init__(self,file=None, data=None, z0=None):
        object.__init__(self)
        self.file = file
        if file is not None:
            if data is not None or z0 is not None:
                raise ValueError('data and z0 must be Noene if file is given')
            else:
                self.load(file)
        else:
            if isinstance(data, dict):
                self.data = pd.DataFrame.from_dict(data)
            else:
                self.data = pd.DataFrame(data)
            if z0 is None:
                self.heights = np.zeros(9)*np.nan
            else:
                self.heights = h_eff(10.,z0)
            self.file = None
            self.header = []

# ----------------------------------------------------

def h_eff(has,z0s):
    '''
    calulate effectice anemometer heights for all
    roughness-length classes used in asutal2000
    :param has: height of the wind measurement in m
    :param z0: roughness length at the site of the wind measurement in m
    :return: list of length 9 containing the nine 
        effecive anemometer heights
    :rtype: list(9)
    '''
    z0_vals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 1.5, 2]
    href = 250
    d0s = displacement_factor*z0s
    ps = np.log((has-d0s)/z0s)/np.log((href-d0s)/z0s)
    ha=[]
    for z0 in z0_vals:
        d0 = displacement_factor*z0
        ha.append(d0 + z0*(( href - d0)/z0)**ps)
    return ha

# ----------------------------------------------------

if __name__ == '__main__':
    #    import matplotlib.pyplot as plt
    #    from matplotlib import cm
    logging.basicConfig(level=logging.DEBUG)
    #
    # test axes
    qq=DataFile('../tests/anno95.akterm')
    print(qq.data[['KENN','FF','DD']])
    qq.write('../tests/out.akterm')

    #
    # test 2D
#    dmna=DataFile('../tests/so2-y00a.dmna')
#    blah=dmna.data['con']
#    print(np.shape(blah))
#    print(np.nanmin(blah),np.nanmax(blah))
#    blah[5,10:15,:]=0.
#    plt.contourf(
#                             np.transpose(blah[:,:,0]),
#                             cmap=cm.get_cmap('YlGnBu')
#                             )

#    # test 3D
#    dmna=DataFile('../tests/w1018a00.dmna')
#    blah=np.sqrt( dmna.data['Vx']**2 + dmna.data['Vy']**2 )
#    print(np.shape(blah))
#    print(np.nanmin(blah),np.nanmax(blah))
#    blah[5,5:10,:]=0.
#    plt.contourf(
#                             np.transpose(blah[:,:,4]),
#                             cmap=cm.get_cmap('magma')
#                             )
#

#    # test zeitreihe
#    dmna=DataFile('../tests/zeitreihe.dmna')
#    t=dmna.data['te']
#    blah=dmna.data['ua']
#    print(np.shape(blah))
#    print(np.nanmin(blah),np.nanmax(blah))
#    plt.plot(t,blah)