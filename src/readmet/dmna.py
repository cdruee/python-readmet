#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The classes and functions in this category handle files
in format "DMNA"
created by Ingenieurbüro Janicke GbR, Überlingen, Germany
(https://www.janicke.de)

The most comprehensive description of this format can be found
in the manual to the `AUSTAL2000 <http://www.austal2000.de>`_
atmospheric dispersion model [JAN2011]_.
"""
import os.path
import re
import struct
import gzip
import logging
import numpy as np
import pandas as pd
from csv import QUOTE_NONE
#
#
#
_KNOWN_TYPES = ['c', 'd', 'x', 'f', 'e', 't']
_IMPLEMENTED_TYPES = ['d', 'f', 'e', 't']
_KNOWN_KEYS = ["cset", "prgm", "artp", "axes", "idnt",
               "t1", "t2", "dt", "dtbnummax", "index",
               "groups", "xmin", "ymin", "delta", "refx",
               "refy", "ggcs", "zscl", "sscl", "sk",
               "uref", "dref", "mode", "avmean",
               "name", "unit", "vldf", "valid", "locl",
               "form", "refv", "exceed", "sequ", "file",
               "dims", "size", "lowb", "hghb"]
_COMPRESSION_LEVEL = 6
# binary format strings:
# assemple binary format
_BINT = {'c': 'c', 'd': 'i', 'hd': 'h', 'x': 'i', 'hx': 'h',
        'f': 'f', 'lf': 'd', 'e': 'f', 'le': 'd', 't': 'i',
        'lt': 'f', }
_BINL = {'c': 1, 'd': 4, 'hd': 2, 'x': 4, 'hx': 2,
        'f': 4, 'lf': 8, 'e': 4, 'le': 8, 't': 4, 'lt': 8, }
# numberformat to read: '<'=little endian 'f'=float


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


def _simplify_form(fmt):
    if "%" not in fmt:
        raise ValueError('not a valid from string: %s' % fmt)
    res = '%'+fmt.split('%')[1]
    for k, v in {'lf': 'f', 'le': 'e', 'hd': 'd', 'he': 'e'}.items():
        res = res.replace(k, v)
    return res

def _to_3d(arr):
    dims = len(np.shape(arr))
    if dims == 1:
        res = arr[:, np.newaxis, np.newaxis]
    elif dims == 2:
        res = arr[:, :, np.newaxis]
    elif dims == 3:
        res = arr
    else:
        raise ValueError('illegal number of dimensions: %d', dims)
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
    ''' name of file loaded into object '''
    text = None
    ''' text contents the file loaded with the (decompressed)
      text contents of an eventual external `datfile` appended '''
    header = dict()
    ''' dictionary containing the dmna header entries as strings'''
    data_file = None
    ''' filename if the data block is stored in a separate file '''
    binary = False
    ''' if data block is text of binary data '''
    compressed = False
    ''' If data block is compressed with gz '''
    filetype = None
    ''' `grid` or `timeseries` '''
    dims = 0
    ''' Number of dimensions '''
    vars = None
    ''' Number of variables in file  '''
    shape = None
    ''' Shape of data files in `data`  '''
    variables = None
    ''' variable names '''
    data = None
    ''' dictionary containing the data from the file loaded.
      The keys are the variable names.
      The values are of type ``pandas.DataFrame`` with time as index,
      if the file contains timeseries.
      The values are of type ``numpy.array`` with time as index,
      if the file contains gridded data. '''
    _locl = "C"
    _variable_type = dict()

    # ----------------------------------------------------------------------
    #
    # functions
    #
    # ----------------------------------------------------------------------
    def _build(self, values=None, axes=None, name=None, types=None,
               vldf="V", origin=None, binary=False, compressed=False):
        """
         generate object from data

         Parameters
         ----------
         values : TYPE, optional
             DESCRIPTION. The default is None.
         axes : TYPE, optional
             DESCRIPTION. The default is None.
         name : TYPE, optional
             DESCRIPTION. The default is None.
         types : TYPE, optional
             DESCRIPTION. The default is None.
         vldf : TYPE, optional
             DESCRIPTION. The default is "V".
         origin : TYPE, optional
             DESCRIPTION. The default is None.
         binary : TYPE, optional
             DESCRIPTION. The default is None.
         compressed : TYPE, optional
             DESCRIPTION. The default is False.

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
            if any(isinstance(x, np.ndarray) for x in values.values()):
                raise ValueError('values elements have wrong type')
            self.filetype = 'grid'
            #
            #  break down axis values
            #
            if axes is not None:
                self._set_axes(axes)
            #
            # cast all matrices to three dimensions
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
                #
                # raise number of dims to three
                #
                values[var] = _to_3d(values[var])
                #
                #  remember dimensions
                #
                if ('dims' in self.header.keys and
                        not self.header['dims'] is None and
                        self.header['dims'] != dims):
                    raise ValueError('data have %d dimensions, '
                                     'but dims is already set to %d' %
                                     (dims, self.header['dims']))
                else:
                    self.header['dims'] = dims
        elif isinstance(values, pd.DataFrame):
            self.filetype = 'timeseries'
            values['te'] = pd.to_datetime(values['te'],
                                          format="  %Y-%m-%d.%H:%M:%S",
                                          utc=True)
        else:
            raise ValueError('dont know how to handle value class: %s' %
                             type(values))
        #
        # determine variable format and range
        #
        form = dict()
        size = -1
        for var in self.variables:
            #
            # auto-determine variable type and field width
            #
            if var not in types.keys():
                if var == "te":
                    self._variable_type[var] = "t"
                elif self.filetype == 'timeseries' and '.' in var:
                    # prefer exp form for source strenghts in timeseries
                    self._variable_type[var] = "e"
                elif all((values[var] - np.floor(values[var])) == 0):
                    self._variable_type[var] = "d"
                else:
                    digits = np.max(np.ceiling(np.log10(np.abs(values[var]))))
                    if digits > 7 or digits < 0:
                        self._variable_type[var] = "e"
                    else:
                        self._variable_type[var] = "f"
            #
            # determine variable format
            #
            if self._variable_type[var] == "d":
                digits = np.max(np.ceiling(np.log10(np.abs(values[var]))))
                digits = max(digits, 4)
                fmt = '%%%dhd' % digits
                flen = digits
            elif self._variable_type[var] == "f":
                digits = np.max(np.ceiling(np.log10(np.abs(values[var]))))
                if all(self._variable_type[var] - np.floor(self._variable_type[var]) == 0):
                    precision = 0
                    digits = max(digits, 5)
                else:
                    precision = 1
                    digits = max(digits + 2, 7)
                fmt = '%%%d.%df' % (digits, precision)
                flen = digits
            elif self._variable_type[var] == "e":
                fmt = "%10.3e"
                flen = 10
            elif self._variable_type[var] == "t":
                fmt = "%20lt"
                flen = 0
            else:
                raise ValueError('wrong type for matrix: %s' % self._variable_type[var])

            form[var] = '%s%s' % (var.lower, fmt)
            size = size + 1 + flen

        #
        # assemble header info
        #
        if not isinstance(binary, bool):
            raise ValueError('binary must be either True or False')
        if binary:
            self.header['mode'] = "binary"
        else:
            self.header['mode'] = "text"
        if not isinstance(compressed, bool):
            raise ValueError('compressed must be either True or False')
        if compressed:
            self.header['cmpr'] = _COMPRESSION_LEVEL
        else:
            self.header['cmpr'] = 0
        if origin is not None:
            if not all(np.isfinite(origin)):
                raise ValueError('origin must be tuple (gx, gy) of numeric')
            gx, gy = origin
            self.header['gx'] = gx
            self.header['gy'] = gy

        self.header['cset'] = "UTF-8"
        self.header['form'] = [form[x] for x in self.variables]
        if self.filetype == 'grid':
            self.header['dims'] = 3
            self.header['lowb'] = [1, 1, 1]
            self.header['hghb'] = self.dims
            self.header['sequ'] = "k+,j-,i+"
            self.header['size'] = size

        elif self.filetype == 'timeseries':
            self.header['dims'] = 1
            self.header['lowb'] = 1
            self.header['hghb'] = len(values.index)
            self.header['sequ'] = "i+"
            self.header['artp'] = "ZA"
        # store data in object
        self.data = values

    def _write_file(self, filename=None):
        """
        write file

        :param filename: defaults to None
        """
        #
        #  write file
        #
        logging.info('writing dmna: %s' % filename)
        #
        #  consistency check
        #
        if set(self.variables) != set(self.data.keys()):
            raise ValueError('variable names do match data dict keys')
        valforms = self._attrib('form')
        logging.debug('valforms: '+str(valforms))
        if isinstance(valforms, str):
            valforms = [valforms]
        (valnams, _, _, _, valspecs) = self._parse_form(valforms)

        if valnams != self.variables:
            raise ValueError('variable names do match format strings')
        dims = self._attrib('dims')
        sequ = self._attrib('sequ')
        lowb = self.header['lowb']
        hghb = self.header['hghb']
        ipos, idir, ilen = self._parse_sequ(dims, sequ, lowb, hghb)
        #
        #  check supported modes
        #
        if filename is None:
            filename = self._attrib('file')
        if not filename.endswith('.dmna'):
            filename = filename + '.dmna'
        self.header['file'] = os.path.splitext(
            os.path.basename(filename))[0]
        logging.debug('writing header to file: %s' % filename)
        data_file, gz = self._get_datfile(filename)
        logging.debug('writing data to file: %s' % data_file)
        mode = self._attrib('mode', 'text')
        #
        # open files
        #
        con1 = open(filename, "w")
        if filename != data_file:
            cmpr = self._attrib('cmpr', None)
            if mode == 'binary':
                filemode="wb"
            else:
                filemode="w"
            if cmpr is None or cmpr == 0:
                con2 = open(data_file, filemode)
            else:
                con2 = gzip.open(data_file, filemode, compresslevel=6)
        else:
            con2 = con1
        #
        # write header
        #
        # loop over known keys but write only keys defined in object
        lines = []
        for key in _KNOWN_KEYS:
            if key in self.header.keys():
                value = self._attrib(key)
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
                logging.debug('header: %s' % lines[-1])
        con1.writelines([x + '\r\n' for x in lines])
        con1.writelines(['*' + '\r\n'])
        #
        # write data body (type specific)
        #
        if self.filetype == 'grid':
            logging.debug('writing fiel type: grid')
            values = [self.data[x] for x in self.variables]
        elif self.filetype == 'timeseries':
            logging.debug('writing file type: timeseries')
            values = [self.data[x].to_numpy() for x in self.variables]
            pass
        nval = len(self.variables)
        #
        # ensure data have three dimensions
        #
        values = [_to_3d(x) for x in values]
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
        if len(reverse_ipos) < 3:
            for x in range(len(reverse_ipos), 3):
                reverse_ipos.append(x)
        out_values=[]
        for nv in range(nval):
            # reorder axes according to "sequ" parameter
            out_values.append(
                np.transpose(values[nv], axes=reverse_ipos))
            out_shape = np.shape(out_values[-1])
        del(values)
        logging.debug('out_values shape: %s' % str(out_shape))
        #
        #  text mode
        if mode == 'text':
            logging.debug('writing file mode: text')
            #
            # write block for each layer (3. dim)
            for layer in range(out_shape[0]):
                #
                # block separator
                if layer > 0 and out_shape[1] > 1:
                    con1.writelines(['*' + '\r\n'])
                #
                # write lines for each y grid line (2. dim)
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
                            except Exception as e:
                                raise ValueError('cannot convert: %s' % format(value))
                            groups.append(field)
                    line = '  '+' '.join(groups)
                    con2.writelines(line + '\r\n')
        elif mode == 'binary':
            logging.debug('writing file mode: binary')
            ## put all values in big number stream
            numrec = np.size(out_values[0])
            numbers = np.full([nval * numrec], fill_value=np.nan)

            for nv in range(nval):
                # serialize data in array
                # in FORTRAN order i.e. last index is counting fastest
                vn = np.reshape(out_values[nv],
                                newshape=[np.size(out_values[nv])],
                                order='C')
                # put all values in one long array
                for x,v in enumerate(vn):
                    numbers[nv + x * nval] = v

            binf = "<" + "".join(_BINT[valspecs[nl]]
                                 for nl in range(nval))
            # write binary data into list
            for nr in range(numrec):
                v = numbers[(nr * nval):((nr+1) * nval)]
                con2.write(struct.pack(binf, *v))

        else:
            raise ValueError('oups! unknown mode in _write' )
        #
        # write footer
        if mode == 'text':
            con2.writelines(['***' + '\r\n'])
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
            logging.debug('divider: {}'.format(divider))
        except ValueError:
            raise RuntimeError("{} is not in DMNA format".format(self.file))
        #
        # convert the file header into named list
        #
        # remove empty lines
        # remeber: The empty string is a False value.
        header_lines = [x.strip()
                        for x in self.text[0:divider] if not x.strip() == '']
        # convert space behind line tag into tab (if not already present)
        header_lines = [re.sub('\\ +', '\t', x) for x in header_lines]
        logging.debug([x for x in header_lines])
        # 1st field is name 2nd and on is content
        header = dict([x.split('\t', 1) for x in header_lines])
        # remove tabs and quotes
        header = {x: re.sub("\t", " ", y) for x, y in header.items()}
        header = {x: re.sub("\\\"", "", y) for x, y in header.items()}
        # append number of header lines in file
        header['lines'] = divider

        for k, v in header.items():
            logging.debug('{:6s} {}'.format(k, v))
        return header

    # ----------------------------------------------------------------------
    #
    # safely get header value
    #
    def _attrib(self, key, default='_fail_on_error_'):
        """
        return value(s) of header item
        :param:key: Name ofe header item to collect
        :param:default: (optional) Value that is returned if the item is
          not found in the header. if `default` is not supplied and
          `key` is not found among the header items. ``ValueError``
          is raised
        :returns: header value(s)
        :rtype: array
        """
        # get localization, use "C" as default while botstrapping
        try:
            locl = self._locl
        except AttributeError:
            locl = 'C'
        logging.debug('looking for key: {}'.format(key))
        if key in self.header.keys():
            # if key is present: use value
            value = self.header[key]
        elif default != '_fail_on_error_':
            # if key is not present and default is set: use default
            value = default
        else:
            # if key is not present and no default is set: fail
            raise ValueError('key "{}" not found in header'.format(key))
        logging.debug('contains value: {}'.format(value))
        if value is None:
            return value
        # split value into space-separated fields
        val = [x.strip() for x in value.split()]
        # try to convert number(s) to numbers
        #    float values to float
        #    integers to integer
        res = []
        for i, v in enumerate(val):
            try:
                v = _locl_float(v, locl)
                if v.is_integer():
                    v = int(v)
                    logging.debug(
                        '... field {:02d} is int  : {:d}'.format(i, v))
                else:
                    logging.debug(
                        '... field {:02d} is float: {:f}'.format(i, v))
            except BaseException:
                logging.debug('... field {:02d} is text : {:s}'.format(i, v))
            res.append(v)
        # if only one value is contained, return as scalar
        if len(res) == 1:
            out = res[0]
        else:
            out = res
        logging.debug('... return ({:s}): {:s}'.format(out.__class__.__name__, str(out)))
        return out

    # ----------------------------------------------------------------------
    #
    # read the actual data from file
    #
    def _parse_form(self, forms):
        """
        parse the format string(s)
        """
        #
        # Format = Format1 Format2 ...
        # Formati = Name%(*Factor)Length.PrecisionSpecifier
        forms = self.header['form'].split(' ')
        nams = []
        facs = []
        lens = []
        prec = []
        specs = []
        for f in forms:
            logging.debug('parsing: "{}"'.format(f))
            if '[' in f:
                raise RuntimeError('repititive format strings are ' +
                                   'not supported by this version')
            #     '
            # Name
            # Name des Datenelementes (optional).
            if '%' in f:
                x, f = f.split('%')
                logging.debug('... name  : "{}"'.format(x))
            else:
                x = ''
            nams.append(x)
            #
            # Factor
            # Skalierungsfaktor (optional einschl. Klammern).
            if ')' in f:
                x = re.sub(r'\(\*(.*)\).*', r'\1', f)
                f = re.sub(r'.*\)', r'', f)
                logging.debug('... factor: "{}"'.format(x))
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
            logging.debug('... length: "{}"'.format(x))
            lens.append(x)
            #
            # Precision
            # Anzahl der Nachkommastellen (bei float-Zahlen).
            x = re.sub(r'^([0-9]*).*', r'\1', f)
            f = re.sub(r'^([0-9]*)', r'', f)
            if x != '':
                x = int(float(x))
                logging.debug('... precis: "{}"'.format(x))
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
            # t    integer    4     Binär:Zeitangabe (ohne Datum):
            #                         vergangene Sekunden
            #                       Text: dd.hh:mm:ss oder hh:mm:ss
            # lt   float      8     Binär: Zeitangabe mit Datum:
            #                         Vorkommastellen: Anzahl der Tage seit
            #                           1899-12-30.00:00:00 plus 106
            #                         Nachkommastellen:
            #                           vergangene Sekunden an diesem Tag
            #                       Text: yyyy-mm-dd.hh:mm:ss
            if f in ['c', 'd', 'hd', 'x', 'hx',
                     'f', 'lf', 'e', 'le', 't', 'lt', ]:
                logging.debug('... specif: "{}"'.format(f))
                specs.append(f)
            else:
                raise IOError('unknown format specifier {}'.format(f))

        return (nams, facs, lens, prec, specs)

    # ----------------------------------------------------------------------
    def _set_axes(self, axes):
        """
        set grid-defining header values from axes or grid tuple
        """
        if not isinstance(axes, pd.DataFrame):
            raise ValueError('axes must be pandas.DataFrame')
        if 'x' not in axes.keys:
            raise ValueError('axes must contain at least `x`')
        if 'y' not in axes.keys:
            dims = 1
            delta = set(np.diff(axes['x']))
            xmin = np.min(axes['x'])
            ymin = None
        else:
            dims = 2
            delta = set().union(
                [np.diff(axes['x']), np.diff(axes['y'])])
            xmin = np.min(axes['x'])
            ymin = np.min(axes['y'])
        if len(delta) > 1:
            raise ValueError('horizontal grid spacing not unique')
        if dims == 2 and 'sk' in axes.keys:
            dims = 3
            sk = axes['sk']
        elif 'z' in axes.keys:
            dims = 3
            sk = axes['Z']
        else:
            sk = None
        #
        #  look for conflicts
        #
        if ('dims' in self.header.keys and
                not self.header['dims'] is None and
                self.header['dims'] != dims):
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
        xx = yy = zz = [0.]
        xmin = ymin = 0.
        xlen = ylen = 0
        #
        # get axis start and length
        #
        dims = int(self._attrib('dims'))
        zlen = 1
        if dims >= 1:
            xlen = self.shape[0]
            xmin = self._attrib('xmin')
        if dims >= 2:
            ylen = self.shape[1]
            ymin = self._attrib('ymin')
        if dims >= 3:
            zlen = self.shape[2]
        sk = self._attrib('sk', '')
        #
        # get spacing
        delta = float(self._attrib('delta'))
        #
        # calculate values
        xx = [xmin + delta * i for i in range(xlen)]
        yy = [ymin + delta * i for i in range(ylen)]
        if sk is not None:
            zz = [float(x) for x in sk]
        else:
            if zlen == 1:
                zz = [0.]
            else:
                raise IOError('file does not contain level ' +
                              'heights: {}'.format(self.file))
        #
        # make dict and return it completely or just one dimension
        axs = {'x': xx, 'y': yy, 'z': zz}
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
            data_file = self._attrib('data_file', None)
        else:
            data_file = None
        # ascii or binary ?
        mode = self._attrib('mode', 'text')
        logging.debug('mode: {}'.format(mode))
        # compression strentgth ?
        cmpr = self._attrib('cmpr', '0')
        logging.debug('cmpr: {}'.format(cmpr))
        #
        # name of separate datafile (if any)
        if data_file is None:
            if mode == 'text' and cmpr > 0:
                data_file = re.sub(r'.dmna$', '.dmnt.gz', filename)
            elif mode == 'text' and cmpr == 0:
                data_file = filename
            elif mode == 'binary' and cmpr == 0:
                data_file = re.sub(r'.dmna$', '.dmnb', filename)
            elif mode == 'binary' and cmpr > 0:
                data_file = re.sub(r'.dmna$', '.dmnb.gz', filename)
            else:
                raise('illegal data file mode/compression: %s/%s' %
                      (mode, str(cmpr)))
        logging.debug('datfile: {}'.format(data_file))
        if cmpr > 0:
            gz = True
        else:
            gz = False
        return (data_file, gz)

    def _parse_sequ(self, dims, sequ, lowb, hghb):
        """
        get index oder and orientation
        index sequence gives order (slowest counting to fastest counting)
        of numbers in file e.g. "k+,j-,i+"
        index position is position of axis in list seq
        e.g. x-axis boundaries are in first column in lowb/highb
             x-index "i" is found in last position, direction is +
                    -> fastest counting, increasing
                    -> along data rows, lowes x left highest x right
        """
        logging.debug('sequ: {}'.format(sequ))
        sequ = sequ.split(',')
        if len(sequ) != dims:
            print(sequ, len(sequ), dims, len(sequ) - dims)
            raise IOError('number of indices does not match number of' +
                          ' dimensions in: {}'.format(self.file))
        if dims in [1, 2, 3]:
            # index names
            inam = ['i', 'j', 'k']
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
            raise IOError(
                '{} dimensions are not supported by this version'.format(dims))
        #
        # index boundaries
        #
        lowb = [int(x) for x in self.header['lowb'].split()]
        hghb = [int(x) for x in self.header['hghb'].split()]
        ilen = [x - y + 1 for x, y in zip(hghb, lowb)]

        logging.debug('ipos:   {}'.format(ipos))
        logging.debug('idir:   {}'.format(idir))
        logging.debug('ilen:   {}'.format(ilen))
        return ipos, idir, ilen

    # ----------------------------------------------------------------------
    #
    # read variable definitions
    #
    def _get_data(self):
        """
         parses the header dictionary and gets number and kind of dimensions
         """
        dims = self._attrib('dims')
        #
        # get index oder and orientation
        #
        if ('sequ' not in self.header.keys() or
                'lowb' not in self.header.keys() or
                'hghb' not in self.header.keys()):
            raise IOError('file does not contain information ' +
                          'on grid size: {}'.format(self.file))
        sequ = self._attrib('sequ')
        lowb = self.header['lowb']
        hghb = self.header['lowb']
        ipos, idir, ilen = self._parse_sequ(dims, sequ, lowb, hghb)
        #
        # how many values per data record
        #
        form = self._attrib('form', None)
        if form is not None:
            (valnams, valfacs, vallens, valprec, valspec
             ) = self._parse_form(form)
            nval = len(valspec)
        else:
            nval = 1

        logging.debug('nval:   {}'.format(nval))
        logging.debug('valnams : {}'.format(valnams))
        logging.debug('valfacs : {}'.format(valfacs))
        logging.debug('vallens : {}'.format(vallens))
        logging.debug('valprec : {}'.format(valprec))
        logging.debug('valspecc: {}'.format(valspec))

        #
        # ascii or binary ?
        mode = self._attrib('mode', 'text')
        logging.debug('mode: {}'.format(mode))
        #
        # number format ?
        locl = self._attrib('locl', 'C')
        logging.debug('locl: {}'.format(mode))
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
                with ofct(self.file, 'r') as file:
                    for x in file.readlines():
                        self.text.append(str(x).rstrip('\n'))
                startline = 0
            else:
                startline =  self.header['lines'] + 1
            # read starting after header plus '*' line:
            for layer in range(numlayer):
                for nl, line in enumerate(self.text[startline:]):
                    if '*' in line:
                        logging.debug(
                            'stopped reading at line {} ("{}")'.format(nl, line))
                        break
                    elif line.strip() != '':
                        for nf, field in enumerate(line.strip().split()):
                            spec = valspec[nf % len(valspec)]
                            if spec in ['c']:
                                x = field
                            elif spec in ['d', 'hd', 'x', 'hx']:
                                x = int(_locl_float(field, locl))
                            elif spec in ['f', 'lf', 'e', 'le']:
                                x = _locl_float(field, locl)
                            elif spec in ['t']:
                                # dd.hh:mm:ss oder hh:mm:ss
                                if '.' in field:
                                    x = np.timedelta64(int(field.split('.')[0]), 'D')
                                else:
                                    x = np.timedelta64(0, 's')
                                x = x + (np.datetime64('2000-01-01 ' + field) -
                                         np.datetime64('2000-01-01 00:00:00'))
                            elif spec in ['lt']:
                                # yyyy-mm-dd.hh:mm:ss
                                x = np.datetime64(field.replace('.', ' '))
                            else:
                                raise RuntimeError('internal: illegal format ' +
                                                   'specifier: {}'.format(spec))
                            numbers.append(x)

        elif mode == 'binary':
            binf = "<" + "".join(_BINT[valspec[nl]] for nl in range(nval))
            binl = sum(_BINL[valspec[nl]] for nl in range(nval))
            # read binary data into list
            with ofct(self.data_file, "rb") as ff:
                for nl in range(numrec):
                    numbers += list(struct.unpack(binf, ff.read(binl)))
        else:
            raise IOError('unsupported mode: {}'.format(mode))

        logging.debug('numrec : {}'.format(numrec))
        logging.debug('#values: {}'.format(numrec * nval))
        logging.debug('#read  : {}'.format(len(numbers)))

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
            if self.header['artp'] == 'ZA' and 'te' in out.columns:
                out.loc[:, 'te'] = pd.to_datetime(out['te'])
                out.set_index(out['te'])
        else:
            out = {k: v for k, v in zip(valnams, values)}
        return (dims, nval, ilen, valnams, out)

    # ----------------------------------------------------------------------
    #
    # read file into memory
    #
    def load(self, file, text=False):
        """
        loads the contents of a dmna file into the object

        :param file: filename (optionally including path). \
          If missing, an emtpy
        :param text: (optional) If ``True`` the raw file contents \
          are containted as atrribute `text` in the object. If ``False`` \
          or missing, the raw file contents are discarded after parsing.
        """
        with open(self.file, 'r') as f:
            self.text = [str(x).rstrip('\n') for x in f.readlines()]
        self.header = self._get_header()
        (self.data_file, self.compressed) = self._get_datfile()
        (self.dims, self.vars, self.shape,
         self.variables, self.data) = self._get_data()
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
    # constructor
    #
    def __init__(self, file=None, values=None, axes=None,
                 name=None, types=None, vldf="V",
                 text=None, compressed=False):
        object.__init__(self)
        self.file = file
        if file is not None:
            if all([x is None
                    for x in [values, axes, name, types]]):
                self.load(file, text)
            else:
                raise ValueError('DataFile initialization from file'
                                 ' and from data are mutually exclusive')
        else:
            self._build(values, axes, name, types, vldf, text, compressed)

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
        self._get_axes(ax)

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
        xmin = self._attrib('xmin')
        ymin = self._attrib('ymin')
        delta = self._attrib('delta')
        #
        # reference position
        refx = self._attrib('refx', None)
        refy = self._attrib('refy', None)
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
        '''
        Writes DataFile object to file

        :param filename: (string) name of file to write, optionally
                         containing a path.
        '''
        self._write_file(filename)

