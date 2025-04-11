#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The classes and functions in this category handle files
in format "TOA5"
created by Campbell Scientific (R), Logan, Utah, USA
(https://www.campbellsci.com)

The most comprehensive description of this format can be found
in the LoggerNet sofatware manual from Campbell Scientific,
e.g. Version 4.1 [CS2011]_.
"""

import re
import csv
import pandas as pd
import logging
import numbers

logger = logging.getLogger(__name__)

class Header(dict):
    """
    object class that holds metadate contained in the TOA5 header

    can be accessed like ``dict``
    """

    def __init__(self):
        super().__init__()
        super(Header, self).__setitem__('station_name', '')
        super(Header, self).__setitem__('logger_name', '')
        super(Header, self).__setitem__('logger_serial', '')
        super(Header, self).__setitem__('logger_os', '')
        super(Header, self).__setitem__('logger_prog', '')
        super(Header, self).__setitem__('logger_sig', '')
        super(Header, self).__setitem__('table_name', '')
        super(Header, self).__setitem__('column_names', {})
        super(Header, self).__setitem__('column_units', {})
        super(Header, self).__setitem__('column_sampling', {})

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError("%s is not a legal key of Header" % (repr(key)))
        super(Header, self).__setitem__(key, value)

    def __delitem__(self, key):
        raise KeyError("%s cannot be deleted from Header" % (repr(key)))


def getfields(line):
    """
    parses a line from the file and splits it into fields

    :param line: file line as string (optionally including a trailing newline)

    :returns: field value(s)
    :rtype: ``list`` of ``string``
    """
    line = line.replace('\n', '')
    fields = line.split(',')
    for i in range(0, len(fields)):
        fields[i] = fields[i].replace('"', '')
    return fields


def check_file(filename):
    """
    checks is the given file exists, is readable, and is a TOA5 file

    :param filename: file name (optionally including path)
    :returns: ``True`` if file is valid TOA5, ``False`` else.
    :rtype: ``bool``
    """
    import os.path
    # return values:
    #   1: file is valid TOA5
    #   0: file is not TOA5
    #  -1: file not found
    #   2: read error
    result = 1
    if os.path.isfile(filename):
        try:
            with open(filename, 'r') as fid:
                # read header line 1
                fields = getfields(fid.readline())
        except IOError:
            result = 2  # = False
        if result == 1 and fields[0] != 'TOA5':
            logger.debug('expected "TOA5", got %s' % fields[0])
            result = 0  # = False
    else:
        result = -1  # = False
    return result


def get_header(filename):
    """
    extracts header from the file

    :param filename: file name (optionally including path)

    :returns: header from file given
    :rtype: ``Header``
    """
    header = Header()
    with open(filename, 'r') as fid:
        # read header line 1
        fields = getfields(fid.readline().rstrip())
        header['station_name'] = fields[1]
        header['logger_name'] = fields[2]
        header['logger_serial'] = fields[3]
        header['logger_os'] = fields[4]
        header['logger_prog'] = fields[5]
        header['logger_sig'] = fields[6]
        header['table_name'] = fields[7]
        # read header line 2
        columns = getfields(fid.readline().rstrip())
        columnnames = []
        for col in columns:
            columnnames.append(re.sub('[()]', '_', col))
        header['column_names'] = columnnames
        # read header line 3
        header['column_units'] = getfields(fid.readline().rstrip())
        # read header line 4
        header['column_sampling'] = getfields(fid.readline().rstrip())
    return header


def get_data(filename):
    """
    extracts data table from the file

    :param filename: file name (optionally including path)

    :returns: data from the file given
    :rtype: ``pandas.DataFrame``
    """
    header = get_header(filename)
    data = pd.read_csv(filename, header=None, skiprows=4, sep=',',
                       quotechar='"', na_values=['NAN', '+INF', '-INF'])
    data.columns = header['column_names']
    # if there are fractional seconds, make all timestamps have them
    if any("." in x for x in data['TIMESTAMP']):
        fmt = '%Y-%m-%d %H:%M:%S.%f'
    else:
        fmt = '%Y-%m-%d %H:%M:%S'
    data.index = pd.to_datetime(data['TIMESTAMP'], format=fmt)
    return data


def verify_header(header):
    hdr = Header()
    for f in hdr.keys():
        if f not in header.keys():
            raise ValueError('header field {:s} is missing')
    ln = len(header['column_names'])
    lu = len(header['column_units'])
    ls = len(header['column_sampling'])
    if ln != lu or ln != ls:
        raise ValueError('column header length mismatch')
    return


def write_header(file, header):
    """
    writes header to the file

    :param file: file name (optionally including path)
    :param header: file header information as type ``Header``

    """
    # binary mode needed, see
    # http://stackoverflow.com/questions/18449233/2-7-csv-module-wants-unicode-but-doesnt-want-unicode
    with open(file, 'w') as fid:
        csvwriter = csv.writer(fid,
                               delimiter=',',
                               quotechar='"',
                               quoting=csv.QUOTE_ALL)
        # read header line 1
        fields = ['TOA5',
                  header['station_name'],
                  header['logger_name'],
                  header['logger_serial'],
                  header['logger_os'],
                  header['logger_prog'],
                  header['logger_sig'],
                  header['table_name']]
        csvwriter.writerow([f for f in fields])
        # read header line 2
        csvwriter.writerow([f for f in header['column_names']])
        # read header line 3
        csvwriter.writerow([f for f in header['column_units']])
        # read header line 4
        csvwriter.writerow([f for f in header['column_sampling']])


def format_field(v: float):
    """
    format information in the same way the CampbellScientific software does: \
    10 decimals below 0.001, \
    9 decimals below 0.01, \
    8 decimals below 0.1, \
    7 decimals below 1.0, \
    9 significant digits below 10^9, and\
    no decimals above, \
    stripping all trailing zeroes (!) and trailing decimal points (!).

    :param v: number or string to be formatted

    :returns: formatted string
    :rtype: ``str``

    """
    if isinstance(v, str):
        f = '"{:s}"'.format(v)
    elif isinstance(v, numbers.Number):
        #  elif isinstance(v,float) or isinstance(v,numpy.int64):
        if abs(v) < 0.001:
            f = '{:15.10f}'.format(v)
        elif abs(v) < 0.01:
            f = '{:15.9f}'.format(v)
        elif abs(v) < 0.1:
            f = '{:15.8f}'.format(v)
        elif abs(v) < 1.:
            f = '{:15.7f}'.format(v)
        elif abs(v) < 1000000000.:
            f = '{:9f}'.format(v)
        else:
            f = '{:.0f}'.format(v)
        f = f.strip()
        if '.' in f:
            f = f.rstrip('0')
        f = f.rstrip('.')
    elif isinstance(v, pd.Timestamp):
        f = v.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0.')
    else:
        logger.debug('{}: {}'.format(type(v), v))
        raise TypeError(
            'format_field expects str or number, got %s' %
            format(type(v).__name__)
        )
    return f


def write_data(file, values):
    """
    writes data to the file. Data are appendend to file if file exists,
    leaving a header or other data in place

    :param file: file name (optionally including path)
    :param values: data table as type ``pandas.DataFrame``

    """
    if 'time' in values.columns:
        values = values.drop('time', 1)
    # binary mode needed, see
    # http://stackoverflow.com/questions/18449233/
    #        2-7-csv-module-wants-unicode-but-doesnt-want-unicode
    with open(file, 'ab') as fid:
        for i, r in values.iterrows():
            line = ','.join(format_field(f) for f in r)
            fid.write(line.encode() + b'\n')


def write_file(filename, header, data):
    """
    writes header plus data to the file, without further checks,
    overwriting any information if the file exists.

    :param filename: file name (optionally including path)
    :param header: file header information as type ``Header``
    :param data: data table as type ``pandas.DataFrame``
    """
    write(filename, header, data)


def write(filename, header, data):
    """
    checks, if data and header are consistsnt and then
    writes header plus data to the file,
    overwriting any information if the file exists.

    :param filename: file name (optionally including path)
    :param header: file header information as type ``Header``
    :param data: data table as type ``pandas.DataFrame``
    """
    verify_header(header)
    write_header(filename, header)
    write_data(filename, data)


def read(filename):
    """
    extracts header and data from the file

    :param filename: file name (optionally including path)

    :returns: header and data from file given
    :rtype: ``Header`` and ``panads.DataFrame``
    """
    logger.info("reading file: %s" % filename)
    return get_header(filename), get_data(filename)
