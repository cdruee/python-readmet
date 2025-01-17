#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The classes and functions in this category handle files
in format "DMNA"
created by Ingenieurbüro Janicke GbR, Überlingen, Germany
(https://www.janicke.de)

The most comprehensive description of this format can be found
in the manual to the `AUSTAL2000 <https://www.austal2000.de>`_
atmospheric dispersion model [JAN2011]_.
"""
import codecs
import os.path
import re
import struct
import gzip
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
#
#
#
_KNOWN_TYPES = ['c', 'd', 'x', 'f', 'e', 't']
_IMPLEMENTED_TYPES = ['d', 'f', 'e', 't']
_COMPRESSION_LEVEL = 6
# binary format strings:
# assemble binary format
_BINT = {'c': 'c', 'd': 'i', 'hd': 'h', 'x': 'i', 'hx': 'h',
         'f': 'f', 'lf': 'd', 'e': 'f', 'le': 'd', 't': 'i',
         'lt': 'f', }
_BINL = {'c': 1, 'd': 4, 'hd': 2, 'x': 4, 'hx': 2,
         'f': 4, 'lf': 8, 'e': 4, 'le': 8, 't': 4, 'lt': 8,
         's': 1}
_BINP = {'c': bytes, 'd': int, 'hd': int, 'x': int, 'hx': int,
         'f': float, 'lf': float, 'e': float, 'le': float, 't': int,
         'lt': float, }
# number format to read: '<'=little endian 'f'=float
_ENCODINGS = ['us-ascii', 'iso-8859-1', 'utf-8', ]
_COMMENT_CHAR = "'"


def _locl_float(s, locl):
    """
    converts localized number strings to float.
    German number localization ("Dezimal-Komma") is respected
    depending on the "locl" header parameter.

    """
    if locl == 'C':
        pass
    elif locl == 'german':
        # remove points every three digits:
        s = s.replace('.', ' ')
        # decimal comma -> decimal point
        s = s.replace(',', '.')
    else:
        raise ValueError('unknown locl: "{}"'.format(locl))
    return float(s)


def _parsedifftime(s):
    """
    parse time-delta string of format ddd.hh:mm:ss
    :return: time difference
    :rtype: pandas.Timedelta
    """
    if '.' in s:
        d, p = s.split('.')
    else:
        d, p = '0', s
    return pd.to_timedelta(int(d), 'days') + pd.to_timedelta(p)


def _parse_form(forms):
    """
    parse the format string(s)
    """
    #
    # Format = Format1 Format2 ...
    # Formati = Name%(*Factor)Length.PrecisionSpecifier
    #
    # if forms is a string, find out if it is a single
    # format or a concatenated list of formats
    if isinstance(forms, str):
        if forms.count('%') < 1:
            raise ValueError('error in format string %s' % forms)
        elif forms.count('%') == 1:
            forms = [forms]
        else:
            forms = re.findall(r'[^%]*%[\[\].0-9]+[hl]?[cdxfets]',
                               forms)
    # if one or more formats contain repetition,
    # replace by matching number of single formats:
    forms2 = []
    rep_mark = r'\[[0-9]+\]'
    for f in forms:
        if re.search(rep_mark, f):
            rep_count = int(re.sub(r'.*\[([0-9]+)].*', r'\1', f))
            rep_form = re.sub(r'\[([0-9]+)]', '', f)
            for i in range(rep_count):
                forms2.append(rep_form)
        else:
            forms2.append(f)
    forms = forms2
    del forms2
    #
    # now parse all forms
    #
    nams = []
    facs = []
    lens = []
    prec = []
    specs = []
    nbyte = []
    for f in forms:
        logger.debug('parsing: "{}"'.format(f))
        #     '
        # Name des Datenelementes (optional).
        if '%' in f:
            x, f = f.split('%')
            logger.debug('... name  : "{}"'.format(x))
        else:
            x = ''
        nams.append(x)
        #
        # Factor
        # Skalierungsfaktor (optional einschl. Klammern).
        if ')' in f:
            x = re.sub(r'\(\*(.*)\).*', r'\1', f)
            f = re.sub(r'.*\)', r'', f)
            logger.debug('... factor: "{}"'.format(x))
        else:
            x = '1.0'
        facs.append(float(x))
        #
        # Length
        # Länge des Datenfeldes.
        if '.' in f:
            x = re.sub(r'(.*)\..*', r'\1', f)
            f = re.sub(r'.*\.', r'', f)
        else:
            x = re.sub(r'([0-9]*).*', r'\1', f)
            f = re.sub(r'([0-9]*)', r'', f)
        x = int(float(x))
        logger.debug('... length: "{}"'.format(x))
        lens.append(x)
        #
        # Precision
        # Anzahl der Nachkommastellen (bei float-Zahlen).
        x = re.sub(r'^([0-9]*).*', r'\1', f)
        f = re.sub(r'^([0-9]*)', r'', f)
        if x != '':
            x = int(float(x))
            logger.debug('... precis: "{}"'.format(x))
        else:
            x = None
        prec.append(x)
        #
        # Specifier
        # Umwandlungsangabe.
        # Folgende Umwandlungsangaben sind möglich:
        # Spec. Typ        Bytes Beschreibung
        # c    character  1     einzelne Buchstaben
        # d    integer    4     Dezimalzahl
        # hd   integer    2     Dezimalzahl
        # x    integer    4     Hexadezimalzahl
        # hx   integer    2     Hexadezimalzahl
        # f    float      4     Festkommazahl (ohne Exponent)
        # lf   float      8     Festkommazahl (ohne Exponent)
        # e    float      4     Gleitkommazahl (mit Exponent)
        # le   float      8     Gleitkommazahl (mit Exponent)
        # t    integer    4     Binär: Zeitangabe (ohne Datum):
        #                         vergangene Sekunden
        #                       Text: dd.hh:mm:ss oder hh:mm:ss
        # lt   float      8     Binär: Zeitangabe mit Datum:
        #                         Vorkommastellen: Anzahl der Tage seit
        #                           1899-12-30.00:00:00 plus 106
        #                         Nachkommastellen:
        #                           vergangene Sekunden an diesem Tag
        #                       Text: yyyy-mm-dd.hh:mm:ss
        #
        # OR (not mentioned in the list):
        #
        # s    integer    1     used for z0-class in roughness map.
        #                       the length-1 numbers are stored
        #                       without separator. i.e.
        #                       length corresponds to number of
        #                       cells in west-east direction.
        #                       precision seems to be length+1 but
        #                       why is not noted in documentation.
        #
        if f in ['c', 'd', 'hd', 'x', 'hx',
                 'f', 'lf', 'e', 'le', 't', 'lt',
                 's'
                 ]:
            logger.debug('... specif: "{}"'.format(f))
            specs.append(f)
            nbyte.append(_BINL[f])
        else:
            raise IOError('unknown format specifier {}'.format(f))

    return nams, facs, lens, prec, specs, nbyte


def _parse_sequ(dims, sequ, lowb, hghb):
    """
    get index oder and orientation
    index sequence gives order (slowest counting to fastest counting)
    of numbers in file e.g. "k+,j-,i+"
    index position is position of axis in list seq
    e.g. x-axis boundaries are in first column in lowb/highb
         x-index "i" is found in last position, direction is +
                - fastest counting, increasing
                - along data rows, lowes x left highest x right
    """
    logger.debug('sequ: {}'.format(sequ))
    sequ = sequ.split(',')
    if len(sequ) != dims:
        print(sequ, len(sequ), dims, len(sequ) - dims)
        raise ValueError('number of indices does not match number of' +
                         ' dimensions')
    if dims in [1, 2, 3, 4]:
        # index names
        inam = ['i', 'j', 'k', 'l']
        # direction of each index in sequence
        # take second character of sequence entry,
        # assume "+" if 2nd character is missing
        seqind = [x[0] for x in sequ[0:dims]]
        seqdir = [x[1] if len(x) > 1 else '+' for x in sequ[0:dims]]
        # position of each index in sequence
        ipos = [0] * dims
        idir = [''] * dims
        for nl in range(dims):
            if inam[nl] in seqind:
                ipos[nl] = seqind.index(inam[nl])
                idir[nl] = seqdir[ipos[nl]]
    else:
        raise ValueError(
            '{} dimensions are not supported by this version'.format(dims))
    #
    # index boundaries
    #
    if isinstance(hghb, int):
        hghb = [hghb]
    if isinstance(lowb, int):
        lowb = [lowb]
    ilen = [x - y + 1 for x, y in zip(hghb, lowb)]

    logger.debug('ipos:   {}'.format(ipos))
    logger.debug('idir:   {}'.format(idir))
    logger.debug('ilen:   {}'.format(ilen))
    return ipos, idir, ilen


def _simplify_form(fmt):
    if "%" not in fmt:
        raise ValueError('not a valid from string: %s' % fmt)
    res = '%' + fmt.split('%')[1]
    for k, v in {'lf': 'f', 'le': 'e', 'hd': 'd', 'he': 'e'}.items():
        res = res.replace(k, v)
    return res


def _to_3d(arr):
    dims = len(np.shape(arr))
    if dims == 1:
        res = arr[np.newaxis,  :, np.newaxis]
    elif dims == 2:
        res = arr[np.newaxis, :, :]
    elif dims == 3:
        res = arr
    else:
        raise ValueError('illegal number of dimensions: %d', dims)
    return res

def _count_digits(arr: np.ndarray) -> int:
    # this code produces Warnings for 0.0 in arr
    # np.max(np.ceil(np.log10(np.abs(arr))))
    #
    # this code counts actual digits
    bb = arr.copy()
    bb[~np.isfinite(bb)] = 0

    lx = len(str(int(np.max(bb))))
    ln = len(str(int(np.min(bb))))
    res = max(lx, ln)
    return res
#
#
#
# ------------------------------------------------------------------------
#


class DataFile(object):
    """
     object class that holds data and metadata of a dmna file

     :param file: filename (optionally including path). \
       If missing, an emtpy object is returned
     :param text: (optional) If ``True`` the raw file contents \
       are containted as atrribute `text` in the object. If ``False`` \
       or missing, the raw file contents are discarded after parsing.
     """
    file = None
    """ name of file loaded into object """
    text = None
    """ text contents the file loaded with the (decompressed)
      text contents of an eventual external `datfile` appended """
    header = dict()
    """ dictionary containing the dmna header entries as strings"""
    data_file = None
    """ filename if the data block is stored in a separate file """
    binary = False
    """ if data block is text of binary data """
    compressed = False
    """ If data block is compressed with gz """
    filetype = None
    """ `grid` or `timeseries` """
    dims = 0
    """ Number of dimensions """
    vars = None
    """ Number of variables in file  """
    shape = None
    """ Shape of data files in `data`  """
    variables = None
    """ variable names """
    data = None
    """ dictionary containing the data from the file loaded.
      The keys are the variable names.
      The values are of type ``pandas.DataFrame`` with time as index,
      if the file contains timeseries.
      The values are of type ``numpy.array``,
      if the file contains gridded data. """
    _locl = "C"
    _variable_type = dict()

    # ----------------------------------------------------------------------
    #
    # constructor
    #
    def __init__(self, file=None, values=None, axs=None,
                 name=None, types=None,
                 cmpr=False, mode='text',
                 text=False, header_only=False, **kwargs):
        object.__init__(self)
        self.file = file
        if file is not None:
            if all([x is None
                    for x in [values, axs, name, types]]):
                self.load(file, text=text, header_only=header_only)
            else:
                raise ValueError('DataFile initialization from file'
                                 ' and from data are mutually exclusive')
        else:
            self._build(values, axs, name, types, cmpr, mode,
                        **kwargs)

    # ----------------------------------------------------------------------
    #
    # functions
    #
    # ----------------------------------------------------------------------
    def _build(self, values=None, axs=None, name=None, types=None,
               cmpr=False, mode='text', **kwargs):
        """
         generate object from data

         Parameters
         ----------
         values : TYPE, optional
             DESCRIPTION. The default is None.
         axs : TYPE, optional
             DESCRIPTION. The default is None.
         name : TYPE, optional
             DESCRIPTION. The default is None.
         types : TYPE, optional
             DESCRIPTION. The default is None.
         cmpr : TYPE, optional
             DESCRIPTION. The default is False.
         mode : TYPE, optional
             DESCRIPTION. The default is None.
         **kwargs : TYPE, optional
             will be added to the header
         Raises
         ------
         ValueError
            DESCRIPTION.
         number
             DESCRIPTION.

         Returns
         -------
         None.

         """
        if isinstance(values, np.ndarray):
            #
            # cast array to single-element list
            #
            # name is mandatory
            if name is None:
                raise ValueError('name must be given if values is np.ndarray')
            # convert type
            values = {name: values}
        elif isinstance(values, dict) or isinstance(values, pd.DataFrame):
            pass
        else:
            raise ValueError('values has wrong type: %s' % type(values))
        #
        # store variable names
        #
        self.variables = list(values.keys())
        #
        # apply variable types names given separately
        #
        if types is not None:
            if set(types) != set(self.variables):
                raise ValueError('names of types must match '
                                 'names of values')
            for var in self.variables:
                if types[var] not in _KNOWN_TYPES:
                    raise ValueError('unknown type "%s" for variable: %s' %
                                     (types[var], var))
                elif types[var] not in _IMPLEMENTED_TYPES:
                    raise ValueError('type not implemented: %s' %
                                     types[var])
                else:
                    self._variable_type[var] = types[var]
        #
        # type-specific processing
        #
        if isinstance(values, dict):
            if any(not isinstance(x, np.ndarray) for x in values.values()):
                raise ValueError('values elements have wrong type')
            self.filetype = 'grid'
            #
            #  break down axis values
            #
            if axs is not None:
                self._set_axes(axs)
            #
            # # cast all matrices to three dimensions
            #
            global_shape = None
            dims = 0
            for var in self.variables:
                #
                # check that all matrices have same dims
                #
                if global_shape is None:
                    global_shape = values[var].shape
                    dims = len(global_shape)
                else:
                    if np.shape(values[var]) != global_shape:
                        raise ValueError('variable does not have identical '
                                         'shape: %s', var)
                # #
                # # raise number of dims to three
                # #
                # values[var] = _to_3d(values[var])
                #
                #  remember dimensions
                #
                if self._att1('dims', None) is None:
                    self.header['dims'] = format(dims)
                elif self._att1('dims', None) != dims:
                    raise ValueError('data have %d dimensions, '
                                     'but dims is already set to %d' %
                                     (dims, self.header['dims']))
                # else dims matches existing header entry

        elif isinstance(values, pd.DataFrame):
            self.filetype = 'timeseries'
            values['te'] = pd.to_datetime(values['te'],
                                          format="  %Y-%m-%d.%H:%M:%S",
                                          utc=True)
            dims = 1
            global_shape = (len(values.index),)
        else:
            raise ValueError('dont know how to handle value class: %s' %
                             type(values))
        # store data in object
        self.data = values

        #
        # assemble header info
        #
        if mode not in ["binary", "text"]:
            raise ValueError("mode must be `text` or `binary`")
        self.header['mode'] = mode

        if cmpr in [True]:
            self.header['cmpr'] = _COMPRESSION_LEVEL
        elif cmpr in [0, False]:
            self.header['cmpr'] = 0
        elif cmpr in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.header['cmpr'] = cmpr
        else:
            raise ValueError('cmpr must be boolean or between 0 and 9')

        if kwargs is not None and len(kwargs) > 0:
            for k, v in kwargs.items():
                kk = str(k)
                if not kk.isalnum():
                    raise ValueError('name of argument %s'
                                     'is not alphanumeric' % kk)
                if pd.api.types.is_list_like(v):
                    vv = ' '.join([format(x) for x in v])
                elif pd.api.types.is_scalar(v):
                    vv = format(v)
                else:
                    raise ValueError('value of argument %s'
                                     'is not scalar or list-like' % kk)
                self.header[format(kk)] = format(vv)

        if 'cset' not in self.header:
            self.header['cset'] = "UTF-8"
        if 'form' not in self.header:
            form = self._make_form(types)
            self.header['form'] = " ".join(["{:s}".format(form[x])
                                            for x in self.variables])
        if self.filetype == 'grid':
            if "sequ" not in self.header:
                self.header['sequ'] = [None,
                                       "i+",
                                       "j-,i+",
                                       "k+,j-,i+"][dims]
            (_, _, tlen, _, _, blen) = _parse_form(self.header['form'])
            if "size" not in self.header:
                # if self.header['mode'] == 'binary':
                self.header['size'] = sum(blen)
                # elif self.header['mode'] == 'text':
                #     self.header['size'] = sum(tlen) + len(tlen) - 1
                # else:
                #     raise RuntimeError('mising mode')
            # index range
            if "lowb" not in self.header and "hghb" not in self.header:
                # automatic
                self.header['lowb'] = [1, 1, 1][0:dims]
                self.header['hghb'] = global_shape[0:dims]
            elif "lowb" in self.header and "hghb" not in self.header:
                # lowb prescribed
                self.header['hghb'] = [self._attr('lowb')[i] -
                                       1 +
                                       global_shape[i]
                                       for i in range(dims)]
            elif "lowb" not in self.header and "hghb" in self.header:
                # hgh prescribed
                self.header['lowb'] = [self._attr('hghb')[i] +
                                       1 -
                                       global_shape[i]
                                       for i in range(dims)]
            elif "lowb" in self.header and "hghb" in self.header:
                # both prescribed
                if any([(self._attr('hghb')[i] -
                         self._attr('lowb')[i] +
                         1) != global_shape[i] for i in range(dims)]):
                    raise ValueError('values lowb and high do match data')

        elif self.filetype == 'timeseries':
            if "dims" not in self.header:
                self.header['dims'] = 1
            if "sequ" not in self.header:
                self.header['sequ'] = "i+"
            if "artp" not in self.header:
                self.header['artp'] = "ZA"
            # index range
            if "lowb" not in self.header and "hghb" not in self.header:
                # automatic
                self.header['lowb'] = 1
                self.header['hghb'] = len(values.index)
            elif "lowb" in self.header and "hghb" not in self.header:
                # lowb prescribed
                self.header['hghb'] = (self._attr('lowb')[0] +
                                       len(values.index))
            elif "lowb" not in self.header and "hghb" in self.header:
                # hgh prescribed
                self.header['lowb'] = (self._attr('hghb')[0] + 1 -
                                       len(values.index))
            elif "lowb" in self.header and "hghb" in self.header:
                # both prescribed
                if (self._attr('hghb')[0] -
                        self._attr('lowb')[0] + 1) != len(values.index):
                    raise ValueError('values lowb and high do match data')

    def _make_form(self, types):
        #
        # determine variable format and range
        #
        form = dict()
        for var in self.variables:
            #
            # auto-determine variable type and field width
            #
            if types is None or var not in types.keys():
                if var == "te":
                    self._variable_type[var] = "t"
                elif self.filetype == 'timeseries' and '.' in var:
                    # prefer exp form for source strengths in timeseries
                    self._variable_type[var] = "e"
                elif np.all((self.data[var] - np.floor(self.data[var]))
                            == 0):
                    self._variable_type[var] = "d"
                else:
                    digits = _count_digits(self.data[var])
                    if digits > 7 or digits < 0:
                        self._variable_type[var] = "e"
                    else:
                        self._variable_type[var] = "f"
            #
            # determine variable format
            #
            if self._variable_type[var] == "d":
                digits = max(_count_digits(self.data[var]), 4)
                fmt = '%%%dhd' % digits
            elif self._variable_type[var] == "f":
                digits = _count_digits(self.data[var])
                if np.all(self.data[var] - np.floor(self.data[var]) == 0):
                    precision = 0
                    digits = max(digits, 5)
                else:
                    precision = 1
                    digits = max(digits + 2, 7)
                fmt = '%%%d.%df' % (digits, precision)
            elif self._variable_type[var] == "e":
                fmt = "%10.3e"
            elif self._variable_type[var] == "t":
                fmt = "%20lt"
            else:
                raise ValueError('wrong type for matrix: %s' %
                                 self._variable_type[var])

            form[var] = '%s%s' % (var, fmt)

        return form

    def _write_file(self, filename=None):
        """
        write file

        :param filename: defaults to None
        """
        #
        #  write file
        #
        logger.info('writing dmna: %s' % filename)
        #
        #  consistency check
        #
        if set(self.variables) != set(self.data.keys()):
            raise ValueError('variable names do match data dict keys')
        valforms = self._attr('form')
        logger.debug('valforms: ' + str(valforms))
        if isinstance(valforms, str):
            valforms = [valforms]
        (valnams, _, vallens, _, valspecs, valbyte) = _parse_form(valforms)

        if valnams != self.variables:
            raise ValueError('variable names do match format strings')
        dims = self._att1('dims')
        sequ = self._att1('sequ')
        lowb = self._attr('lowb')
        hghb = self._attr('hghb')
        ipos, idir, ilen = _parse_sequ(dims, sequ, lowb, hghb)
        #
        #  check supported modes
        #
        if filename is None:
            filename = self._att1('file')
        if not filename.endswith('.dmna'):
            filename = filename + '.dmna'
        self.header['file'] = os.path.splitext(
            os.path.basename(filename))[0]
        logger.debug('writing header to file: %s' % filename)
        data_file, gz = self._get_datfile(filename)
        logger.debug('writing data to file: %s' % data_file)
        mode = self._att1('mode', 'text')
        #
        # open files
        #
        con1 = open(filename, "w", newline='\r\n')
        if filename != data_file:
            cmpr = self._att1('cmpr', None)
            if mode == 'binary':
                if cmpr is None or cmpr == 0:
                    con2 = open(data_file, "wb")
                else:
                    con2 = gzip.open(data_file, "wb", compresslevel=6)
            else:
                if cmpr is None or cmpr == 0:
                    con2 = open(data_file, "w",
                                newline='\r\n')
                else:
                    con2 = gzip.open(data_file, "w", compresslevel=6,
                                     newline='\r\n')
        else:
            con2 = con1
        #
        # write header
        #
        # loop over known keys but write only keys defined in object
        lines = []
        for key in self.header.keys():
            # header attributes are prefixed with '_'
            if not key.startswith('_'):
                value = self._attr(key)
                # is value a scalar?
                if not isinstance(value, list):
                    value = [value]
                # numbers without quotes
                if all([np.issubdtype(type(x), np.number) for x in value]):
                    value = '  '.join([str(x) for x in value])
                else:
                    # characters surrounded by quotes
                    value = '  '.join(['"%s"' % x for x in value])
                lines.append('  '.join((key, value)))
                logger.debug('header: %s' % lines[-1])
        con1.writelines([x + '\n' for x in lines])
        con1.write('*' + '\n')
        #
        # write data body (type specific)
        #
        if self.filetype == 'grid':
            logger.debug('writing fiel type: grid')
            values = [self.data[x] for x in self.variables]
        elif self.filetype == 'timeseries':
            logger.debug('writing file type: timeseries')
            values = [self.data[x].to_numpy() for x in self.variables]
        else:
            raise ValueError("illegal type: %s" % self.filetype)
        nval = len(self.variables)
        #
        # reverse order of values if an index was counting backwards
        #
        for nl, v in enumerate(values):
            for k, d in enumerate(idir):
                if d == '-':
                    values[nl] = np.flip(values[nl], k)
        # reorder axes according to "sequ" parameter
        # convert individual fields to stream of numbers
        # [1111],[2222],[3333] -> 123123123123
        reverse_ipos = [ipos.index(x) for x in range(len(ipos))]
        #        if len(reverse_ipos) < 3:
        #            for x in range(len(reverse_ipos), 3):
        #                reverse_ipos.append(x)
        out_values = []
        out_shape = ()
        for nv in range(nval):
            # reorder axes according to "sequ" parameter
            out_values.append(
                np.transpose(values[nv], axes=reverse_ipos))
            out_shape = np.shape(out_values[-1])
        del values
        logger.debug('out_values shape: %s' % str(out_shape))
        #
        #  text mode
        if mode == 'text':
            logger.debug('writing file mode: text')
            #
            # ensure shape has len 3
            if len(out_shape) == 1:
                out_shape = (1,) + out_shape +(1,)
            elif len(out_shape) == 2:
                out_shape = (1,) + out_shape
            elif len(out_shape) == 3:
                pass
            else:
                raise ValueError('internal error: illegal len(out_shape)')
            #
            # ensure data have three dimensions
            out_values = [_to_3d(x) for x in out_values]

            # write block for each layer (3. dim)
            for layer in range(out_shape[0]):
                #
                # block separator
                if layer > 0 and out_shape[1] > 1:
                    con2.write('*' + '\n')
                #
                # write lines for each y grid line (2. dim)
                lines=[]
                for nl in range(out_shape[1]):
                    # write group for each x grid line (1. dim)
                    groups = []
                    for nr in range(out_shape[2]):
                        #
                        # write sequence of all variables in each group
                        for nv, spec in enumerate(valspecs):
                            value = out_values[nv][layer, nl, nr]
                            try:
                                if spec in ['c']:
                                    field = value[0]
                                elif spec in ['d', 'hd', 'x', 'hx',
                                              'f', 'lf', 'e', 'le']:
                                    field = (_simplify_form(
                                        valforms[nv]) % value)
                                elif spec in ['t']:
                                    # dd.hh:mm:ss oder hh:mm:ss
                                    field = pd.to_datetime(
                                        value).strftime(
                                        '%d.%H:%M:%S')
                                elif spec in ['lt']:
                                    # yyyy-mm-dd.hh:mm:ss
                                    field = pd.to_datetime(
                                        value).strftime(
                                        '%Y-%m-%d.%H:%M:%S')
                                else:
                                    raise RuntimeError('internal: '
                                                       'illegal format '
                                                       'specifier: '
                                                       '{}'.format(spec))
                            except Exception:
                                raise ValueError('cannot convert: %s' %
                                                 format(value))
                            groups.append(field)
                    lines.append('  ' + ' '.join(groups) + '\n')
                con2.writelines(lines)
        elif mode == 'binary':
            logger.debug('writing file mode: binary')
            # put all values in big number stream
            numrec = np.size(out_values[0])
            numbers = [None] * (nval * numrec)

            for nv in range(nval):
                # serialize data in array
                # in FORTRAN order i.e. last index is counting fastest
                vn = np.reshape(out_values[nv],
                                newshape=[np.size(out_values[nv])],
                                order='C')
                # put all values in one long array
                for i, v in enumerate(vn):
                    # store at the right position and
                    # change to matching python type
                    numbers[nv + i * nval] = _BINP[valspecs[nv]](v)

            # assemble format string
            binf = "<" + "".join(_BINT[valspecs[i]]
                                 for i in range(nval))
            # write binary data into list
            for nr in range(numrec):
                v = numbers[(nr * nval):((nr + 1) * nval)]
                con2.write(struct.pack(binf, *v))

        else:
            raise ValueError('oups! unknown mode in _write')
        #
        # write footer
        if mode == 'text':
            con2.write('***' + '\n')
        if con1 == con2:
            con1.close()
        else:
            con2.close()
            con1.close()

    #
    # read header from file
    #
    def _get_header(self):
        """
        parses the file as text, finds the divider line "*"
        and returns the header as dictionary
        """
        try:
            divider = self.text.index("*")
            logger.debug('divider: {}'.format(divider))
        except ValueError:
            raise RuntimeError("{} is not in DMNA format".format(self.file))
        #
        # convert the file header into named list
        #
        # remove empty lines and comment lines (beginning with "-")
        # remember: The empty string is a False value.
        header_lines = [x.strip()
                        for x in self.text[0:divider]
                        if not x.strip() == '' and not x.startswith('-')]
        # convert space behind line tag into tab (if not already present)
        header_lines = [re.sub(' +', '\t', x) for x in header_lines]
        logger.debug([x for x in header_lines])
        # 1st field is name 2nd and on is content
        header = {}
        for hl in header_lines:
            kv = hl.split('\t', 1)
            if len(kv) < 2:
                logger.warning('error in header line: "%s"' % hl)
            else:
                header[kv[0]] = kv[1]
        # remove tabs and quotes
        header = {x: re.sub("\t", " ", y) for x, y in header.items()}
        header = {x: re.sub("\\\"", "", y) for x, y in header.items()}
        # append number of header lines in file / attribute prefixed by '_'
        header['_lines'] = divider

        for k, v in header.items():
            logger.debug('{:6s} {}'.format(k, v))
        return header

    # ----------------------------------------------------------------------
    #
    # safely get header value
    #
    def _att1(self, key, default: "str | int | None" = '_fail_error_'):
        # same as _attr but return single value as scalar
        res = self._attr(key, default)
        if res is None:
            out = None
        elif len(res) == 1:
            out = res[0]
        else:
            raise ValueError("attribute is not scalar: %s" % key)
        return out

    def _attr(self, key,
              default: "str | int | None" = '_fail_error_'
              ) -> "list | None":
        """
        return value(s) of header item
        :param:key: Name of header item to collect
        :param:default: (optional) Value that is returned if the item is
          not found in the header. if `default` is not supplied and
          `key` is not found among the header items. ``ValueError``
          is raised
        :param:arr: If True do not flatten len-1 array to scalar
        :returns: header value(s)
        :rtype: array
        """
        # get localization, use "C" as default while bootstrapping
        try:
            locl = self._locl
        except AttributeError:
            locl = 'C'
        logger.debug('looking for key: {}'.format(key))
        if key in self.header.keys():
            # if key is present: use value
            value = self.header[key]
        elif default != '_fail_error_':
            # if key is not present and default is set: use default
            if default is not None:
                value = format(default)
            else:
                value = None
        else:
            # if key is not present and no default is set: fail
            raise ValueError('key "{}" not found in header'.format(key))
        logger.debug('contains value: {}'.format(value))
        if value is None:
            return value
        # split value into space-separated fields
        if isinstance(value, str):
            val = [x.strip() for x in value.split()]
        elif not pd.api.types.is_list_like(value):
            val = [value]
        else:
            val = value
        # try to convert number(s) to numbers
        #    float values to float
        #    integers to integer
        res = list()
        for i, v in enumerate(val):
            try:
                if v == 'None':
                    v = None
                    logger.debug('... field {:02d} is None'.format(i))
                else:
                    if pd.api.types.is_numeric_dtype(v):
                        v = float(v)
                    else:
                        v = _locl_float(v, locl)
                    if v.is_integer():
                        v = int(v)
                        logger.debug(
                            '... field {:02d} is int  : {:d}'.format(i, v))
                    else:
                        logger.debug(
                            '... field {:02d} is float: {:f}'.format(i, v))
            except ValueError:
                logger.debug('... field {:02d} is text : {:s}'.format(i, v))
            res.append(v)
        return res

    # ----------------------------------------------------------------------
    #
    # read the actual data from file
    #

    # ----------------------------------------------------------------------
    def _set_axes(self, axes):
        """
        set grid-defining header values from axes or grid tuple
        """
        if not isinstance(axes, dict):
            raise ValueError('axes must be dict')
        if 'x' not in axes.keys():
            raise ValueError('axes must contain at least `x`')
        if 'y' not in axes.keys():
            dims = 1
            delta = set(np.diff(axes['x']))
            xmin = np.min(axes['x'])
            ymin = None
        else:
            dims = 2
            delta = set(np.diff(axes["x"])) & set(np.diff(axes["y"]))
            xmin = np.min(axes['x'])
            ymin = np.min(axes['y'])
        if len(delta) > 1:
            raise ValueError('horizontal grid spacing not unique')
        if dims == 2 and 'sk' in axes.keys():
            dims = 3
            sk = axes['sk']
        elif 'z' in axes.keys():
            dims = 3
            sk = axes['z']
        else:
            sk = None
        #
        #  look for conflicts
        #
        if self._att1('dims', None) not in [None, dims]:
            raise ValueError('dims is already set to %d' %
                             self.header['dims'])
        self.header['delta'] = list(delta)[0]
        self.header['xmin'] = xmin
        self.header['ymin'] = ymin
        self.header['sk'] = sk
        self.header['dims'] = dims

    # ----------------------------------------------------------------------
    def _get_axes(self, ax=None):
        """
        get grid axes positions in model coordinates

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        IOError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        #
        # "empty" values
        #
        axs = {}
        # get spacing
        delta = float(self._att1('delta'))
        #
        # get axis start and length
        #
        dims = int(self._att1('dims'))
        #
        # calculate values
        if dims >= 1:
            xlen = self.shape[0]
            xmin = float(self._att1('xmin', 0))
            xx = [xmin + delta * i for i in range(xlen)]
            axs['x'] = xx
        if dims >= 2:
            ylen = self.shape[1]
            ymin = float(self._att1('ymin', 0))
            yy = [ymin + delta * i for i in range(ylen)]
            axs['y'] = yy
        if dims >= 3:
            if len(self.shape) > 2:
                zlen = self.shape[2]
            else:
                zlen = 1
            sk = self._attr('sk', None)
            if sk is not None:
                zz = [float(x) for x in sk]
            else:
                if zlen == 1:
                    zz = [0.]
                else:
                    raise IOError('file does not contain level ' +
                                  'heights: {}'.format(self.file))
            axs['z'] = zz
        #
        # return complete dict or just one dimension
        if ax is None:
            return axs
        elif ax in ['x', 'y', 'z']:
            return axs[ax]
        else:
            raise ValueError('unknown axis: {}'.format(ax))

    # ----------------------------------------------------------------------
    #
    # determine if data are stored externally
    #
    def _get_datfile(self, filename=None):
        """
         parses the header dictionary and gets number and kind of dimensions


         Returns
         -------
         datfile : TYPE
             DESCRIPTION.
         gz : TYPE
            DESCRIPTION.

         """
        if filename is None:
            filename = self.file
        # ascii or binary ?
        mode = self._att1('mode', 'text')
        logger.debug('mode: {}'.format(mode))
        # compression strength ?
        cmpr = self._att1('cmpr', '0')
        logger.debug('cmpr: {}'.format(cmpr))
        #
        # name of separate datafile (if any)
        if mode == 'text' and cmpr > 0:
            data_file = re.sub(r'.dmna$', '.dmnt.gz', filename)
        elif mode == 'text' and cmpr == 0:
            data_file = filename
        elif mode == 'binary' and cmpr == 0:
            data_file = re.sub(r'.dmna$', '.dmnb', filename)
        elif mode == 'binary' and cmpr > 0:
            data_file = re.sub(r'.dmna$', '.dmnb.gz', filename)
        else:
            raise ('illegal data file mode/compression: %s/%s' %
                   (mode, str(cmpr)))
        logger.debug('datfile: {}'.format(data_file))
        if cmpr > 0:
            gz = True
        else:
            gz = False
        return data_file, gz

    # ----------------------------------------------------------------------
    #
    # read variable definitions
    #
    def _get_data(self):
        """
         parses the header dictionary and gets number and kind of dimensions
         """
        dims = self._att1('dims')
        #
        # get index oder and orientation
        #
        if ('sequ' not in self.header.keys() or
                'lowb' not in self.header.keys() or
                'hghb' not in self.header.keys()):
            raise IOError('file does not contain information ' +
                          'on grid size: {}'.format(self.file))
        sequ = self._att1('sequ')
        lowb = self._attr('lowb')
        hghb = self._attr('hghb')
        ipos, idir, ilen = _parse_sequ(dims, sequ, lowb, hghb)
        #
        # how many values per data record
        #
        form = self._attr('form', None)
        if form is not None:
            (valnams, valfacs, vallens, valprec, valspec, valbyte
             ) = _parse_form(form)
            nval = len(valspec)

            logger.debug('nval:   {}'.format(nval))
            logger.debug('valnams : {}'.format(valnams))
            logger.debug('valfacs : {}'.format(valfacs))
            logger.debug('vallens : {}'.format(vallens))
            logger.debug('valprec : {}'.format(valprec))
            logger.debug('valspec : {}'.format(valspec))
        else:
            nval = 1
            valnams = valspec = [""]
        #
        # ascii or binary ?
        mode = self._att1('mode', 'text')
        logger.debug('mode: {}'.format(mode))
        #
        # number format ?
        locl = self._att1('locl', 'C')
        logger.debug('locl: {}'.format(mode))
        #
        # select file opening function according to compression
        if self.compressed:
            ofct = gzip.open
        else:
            ofct = open

        #
        # read the data
        #
        # number of number-records to read:
        numrec = 1
        for nl in range(dims):
            numrec = numrec * ilen[nl]
        if dims == 3:
            numlayer = ilen[2]
        else:
            numlayer = 1
        #
        # read all numbers as one big sequence
        numbers = []
        if mode == 'text':
            # load data from separate data file into text buffer
            if self.data_file != self.file:
                with ofct(self.data_file, 'r') as file:
                    for nxt in file.readlines():
                        self.text.append(nxt.decode().rstrip('\n'))
            startline = self.header['_lines'] + 1
            # read starting after header plus '*' line:
            for layer in range(numlayer):
                for nl, tl in enumerate(self.text[startline:]):
                    # remove trailing comments
                    line = str(tl).split(_COMMENT_CHAR)[0]
                    if line.startswith('-'):
                        line = ''
                    # parse lines
                    if '*' in line:
                        # stars denote block boundaries
                        logger.debug(
                            'stopped reading at line {} ("{}")'.format(nl, line))
                        break
                    elif line.strip() == '':
                        # ignore empty lines
                        pass
                    else:
                        for nf, field in enumerate(line.strip().split()):
                            spec = valspec[nf % len(valspec)]
                            if spec in ['c']:
                                nxt = field
                            elif spec in ['d', 'hd', 'x', 'hx']:
                                nxt = int(_locl_float(field, locl))
                            elif spec in ['f', 'lf', 'e', 'le']:
                                nxt = _locl_float(field, locl)
                            elif spec in ['t']:
                                # dd.hh:mm:ss oder hh:mm:ss
                                if '.' in field:
                                    nxt = np.timedelta64(int(
                                        field.split('.')[0]), 'D')
                                else:
                                    nxt = np.timedelta64(0, 's')
                                nxt = nxt + (np.datetime64('2000-01-01 ' +
                                                           field) -
                                             np.datetime64(
                                                 '2000-01-01 00:00:00'))
                            elif spec in ['lt']:
                                # yyyy-mm-dd.hh:mm:ss
                                nxt = np.datetime64(
                                    field.replace('.', ' '))
                            elif spec in ['s']:
                                nxt = field.strip('" ')
                            else:
                                raise RuntimeError(
                                    'internal: illegal format ' +
                                    'specifier: {}'.format(spec))
                            # store value(s)
                            if isinstance(nxt, list):
                                numbers.extend(nxt)
                            else:
                                numbers.append(nxt)

        elif mode == 'binary':
            binf = "<" + "".join(_BINT[valspec[nl]] for nl in range(nval))
            binl = sum(_BINL[valspec[nl]] for nl in range(nval))
            # read binary data into list
            with ofct(self.data_file, "rb") as ff:
                for nl in range(numrec):
                    numbers += list(struct.unpack(binf, ff.read(binl)))
        else:
            raise IOError('unsupported mode: {}'.format(mode))

        logger.debug('numrec : {}'.format(numrec))
        logger.debug('#values: {}'.format(numrec * nval))
        logger.debug('#read  : {}'.format(len(numbers)))

        # split variables to individual fields:
        # 123123123123 -> [1111],[2222],[3333]
        values = []
        for nl in range(nval):
            # select all values of variable #i
            vn = np.array([numbers[nl + x * nval] for x in range(numrec)])
            # data in the file are in FORTRAN order i.e. last index is counting
            # fastest
            vr = np.reshape(vn, newshape=[ilen[x] for x in ipos], order='C')
            # reorder axes according to "sequ" parameter
            values.append(np.transpose(vr, axes=ipos))
            del vn, vr
        #
        # reverse order of values if an index was counting backwards
        #
        for nl, v in enumerate(values):
            for k, d in enumerate(idir):
                if d == '-':
                    values[nl] = np.flip(values[nl], k)
        #
        # make output
        #
        if dims == 1:
            out = pd.DataFrame({k: v for k, v in zip(valnams, values)})
            #
            # if timeseries: find time column and convert to POSIXct
            #
            if ('te' in out.columns and
                    self._att1('artp', None) in [None, 'ZA']):
                out.loc[:, 'te'] = pd.to_datetime(out['te'])
                out.set_index(out['te'])
        else:
            out = {k: v for k, v in zip(valnams, values)}
        return dims, nval, ilen, valnams, out

    # ----------------------------------------------------------------------
    #
    # special treatment for axes=ti (monitor point "measurements")
    #
    def _fix_monitor(self, header, data):
        """
        data in case axes=ti is a timeseries
        although described as 2D array (why?)

        :param header: DataFile header dict
        :param data: DataFile data dict (np.array)
        :return: data as timeseries
        :rtype: pandas.DataFrame
        """
        # exit if data do not need fixing
        if (isinstance(data, pd.DataFrame) and
                'te' in data.columns):
            return data
        # do correction
        logger.debug('filetype axes=ti: adding time column')
        dt = _parsedifftime(header['dt'])  # "01:00:00"
        if 't1' in header.keys():
            t1 = _parsedifftime(header['t1'])  # "00:00:00"
        else:
            t1 = pd.Timedelta(0, unit='s')
        if 't2' in header.keys():
            t2 = _parsedifftime(header['t2'])  # "366.00:00:00"
        else:
            # calculate from t1, dt and length of (first element of) data
            t2 = dt * (next(iter(data.values())).shape[0])
        #
        # example formats:
        # "2000-01-01T00:00:00+0100" or
        # "2000-01-01.00:00:00+0100" or
        # "2000-01-01 00:00:00"
        rd = re.sub("([0-9]{4}-[0-9]{2}-[0-9]{2})[ T.]" +
                    "([0-9]{2}:[0-9]{2}:[0-9]{2}).*",
                    "\\1T\\2",
                    header['rdat'])

        rdat = pd.to_datetime(rd, utc=True)
        te = pd.date_range(start=rdat + t1, end=rdat + t2 - dt, freq=dt)
        logger.debug('... %s -- %s' % (te[0].strftime("%F %T"),
                                        te[-1].strftime("%F %T")))
        res = pd.DataFrame({'te': te})
        points = header['mntn'].split()
        for x in data.keys():
            for i in range(data[x].shape[1]):
                name = "%s.%s" % (points[i], x)
                res[name] = data[x][:, i]
        self.data = res

    # ----------------------------------------------------------------------
    #
    # read file into memory
    #
    def load(self, file, text=False, header_only=False):
        """
        loads the contents of a dmna file into the object

        :param file: filename (optionally including path). \
          If missing, an emtpy
        :param text: (optional) If ``True`` the raw file contents \
          are containted as atrribute `text` in the object. If ``False`` \
          or missing, the raw file contents are discarded after parsing.
        :param header_only: (optional) If ``True`` skip loading the data. \
          Useful for fast scanning of file headers.
        """
        logger.info('loading file: %s' % file)
        for en in _ENCODINGS:
            try:
                with codecs.open(self.file, 'r', encoding=en) as f:
                    self.text = []
                    i = 0
                    for x in f.readlines():
                        i += 1
                        self.text.append(str(x).rstrip('\n').rstrip('\r'))
                        if header_only and self.text[-1].strip() == '*':
                            break
                logger.debug('file encoding: %s' % en)
                break
            except UnicodeDecodeError:
                continue
        self.header = self._get_header()
        if not header_only:
            (self.data_file, self.compressed) = self._get_datfile()
            (self.dims, self.vars, self.shape,
             self.variables, self.data) = self._get_data()
            if (self._att1('axes', None) == 'ti' or
                    self._att1('dims', 0) == 1):
                self._fix_monitor(self.header, self.data)
                self.dims = 1
            if (self.dims == 1 and
                    isinstance(self.data, pd.DataFrame)):
                self.filetype = 'timeseries'
            else:
                self.filetype = 'grid'
        if not text:
            del self.text
        self.file = file

    # ----------------------------------------------------------------------
    #
    # calculate x/y/z axes values in model coordinates
    #
    def axes(self, ax=None):
        """
        Return positions of grid lines in model coordinates

        :param ax: (string, optional) name of axis to return. \
                   If missing or `None`, all axes are returned.

        :return: `dict` with axis names as keys, containing\
                 list(s) of positions as values.
        """
        axes = self._att1('axes', None)
        dims = self._att1('dims', -1)
        logger.debug('axes : %s' % format(axes))
        if axes == 'ti' or dims == 1:
            return self.data['te'].values
        elif axes in ['xy', 'xyz', 'xyzs', None]:
            return self._get_axes(ax)

    # ----------------------------------------------------------------------
    #
    # calculate  in Gauss-Krueger coordinates
    #
    def grid(self, what=None):
        """
        calculate grid definition needed for georeferencing
        :returns xlen: number of cells along x-axis
        :returns ylen: number of cells along x-axis
        :returns xll: right-ward position of lower left (southwest) corner
        :returns yll: u-ward position of lower left (southwest) corner
        :returns delta: grid spacing
        """
        if self.file is None:
            raise AttributeError('no file loaded')
        #
        # get axis start and length
        dims = self.dims
        if dims < 2:
            raise ValueError('file must contain at least two dimensions')
        xlen, ylen = self.shape[0:2]
        xmin = self._att1('xmin')
        ymin = self._att1('ymin')
        delta = self._att1('delta')
        #
        # reference position
        refx = self._att1('refx', None)
        refy = self._att1('refy', None)
        if refx is None or refy is None:
            raise ValueError('file does not contain all information on grid')
        #
        # calculate values
        xll = refx + xmin
        yll = refy + ymin
        #
        var = {'xlen': xlen, 'ylen': ylen,
               'xll': xll, 'yll': yll, 'delta': delta}
        if what is None:
            # return dict
            out = var
        else:
            if what in var:
                out = var[what]
            else:
                raise ValueError('unknown grid variable %s' % what)
        return out

    def write(self, filename):
        """
        Writes DataFile object to file

        :param filename: (string) name of file to write, optionally
                         containing a path.
        """
        self._write_file(filename)
