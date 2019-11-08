#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The classes and functions in this category handle files 
in format "DMNA" 
created by Ingenieurbüro Janicke GbR, Überlingen, Germany
(https://www.janicke.de)

The most comprehensive description of this format can be found
in the manual to the `AUSTAL2000 <http://www.austal2000.de>`_ 
atmospheric dispersion model [JAN2011]_.
'''

import re
import struct
import gzip
import logging
import numpy as np
import pandas as pd
#
#
#
def _locl_float(s,locl):
  '''
  converts localized number strings to float.
  German number localization ("Dezimal-Komma") is respected
  depending on the "locl" header parameter.
  
  '''
  if locl=='C':
    pass
  elif locl=='german':
    # remove points every three digits:
    s = s.replace('.',' ')
    # dezimalkomma -> decimal point
    s = s.replace(',','.')
  else:
    raise ValueError('unknown locl: "{}"'.format(locl))
  return(float(s))
#
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
  text = None 
  ''' text contents the file loaded with the (decompressed)
      text contents of an eventual external `datfile` appended '''
  header = None
  ''' dictionary containing the dmna header entries as strings'''
  datfile = None
  ''' filename if the data block is strored in a separate file '''
  compressed = False
  ''' If data block is compressed with gz '''
  dims = None
  ''' Number of dimensions '''
  vars = None
  ''' Number of variables in file  '''
  shape = None
  ''' Shape of data files in `data`  '''
  data = None 
  ''' dictonary containing the data from the file loaded.
      The keys are the variable names. 
      The values are of type ``pandas.DataFrame`` with time as index,
      if the file containes timeseries.
      The values are of type ``numpy.array`` with time as index,
      if the file containes gridded data. '''

  # ----------------------------------------------------------------------
  #
  # read header
  #
  def _get_header(self):
    '''
    parses the file as text, finds the divider line "*"
    and returns the header as dictionary
    '''
    header={}
    try:
      divider=self.text.index("*")
      logging.debug('divider: {}'.format(divider))
    except ValueError:
      raise RuntimeError("{} is not in DMNA format".format(self.file))
    #
    # convert the file header into named list
    #
    # remove empty lines
    # remeber: The empty string is a False value.
    header_lines=[ x.strip() for x in self.text[0:divider] if not x.strip() == '' ]
    # convert space behind line tag into tab (if not already present)
    header_lines=[ re.sub('\ +','\t',x) for x in header_lines ]
    logging.debug([ x for x in header_lines ])
    # 1st field is name 2nd and on is content
    header=dict([x.split('\t',1) for x in header_lines])
    #remove tabs and quotes
    header={ x:re.sub("\t"," ",y ) for x,y in header.items() }
    header={ x:re.sub("\\\"","",y ) for x,y in header.items() }
    #append number of header lines in file
    header['lines']=divider
    
    for k,v in header.items():
      logging.debug('{:6s} {}'.format(k,v))
    return(header)

  # ----------------------------------------------------------------------
  #
  # safely get header value
  #
  def _attrib(self,key,default='_fail_on_error_'):
    '''
    return value(s) of header item
    :param:key: Name ofe header item to collect
    :param:default: (optional) Value that is returned if the item is
      not found in the header. if `default` is not supplied and 
      `key` is not found among the header items. ``ValueError``
      is raised
    :returns: header value(s)
    :rtype: array
    '''
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
      return(value)
    # split value into space-separated fields  
    val = [x.strip() for x in value.split()]
    # try to convert number(s) to numbers
    #    float values to float
    #    integers to integer
    res=[]
    for i,v in enumerate(val):
      try:
        v = _locl_float(v,locl)
        if v.is_integer():
          v = int(v)
          logging.debug('... field {:02d} is int  : {:d}'.format(i,v))
        else:
          logging.debug('... field {:02d} is float: {:f}'.format(i,v))
      except:
          logging.debug('... field {:02d} is text : {:s}'.format(i,v))
      res.append(v)
    # if only one value is containe, return as scalar  
    if len(res) == 1:
      return(res[0])
    else:
      return(res)
  
  # ----------------------------------------------------------------------
  #
  # read the actual data from file
  #
  def _parse_form(self,forms):
    '''
    parse the format string(s)
    '''
    #
    #Format = Format1 Format2 ...
    #Formati = Name%(*Factor)Length.PrecisionSpecifier
    forms=self.header['form'].split(' ')
    nams=[]
    facs=[]
    lens=[]
    prec=[]
    specs=[]
    for f in forms:
      logging.debug('parsing: "{}"'.format(f))
      if '[' in f:
        raise RuntimeError('repititive format strings are not supported by this version')
      #     '
      #Name
      # Name des Datenelementes (optional).
      if '%' in f:
        x,f = f.split('%')
        logging.debug('... name  : "{}"'.format(x))
      else:
        x=''
      nams.append(x)  
      #
      #Factor
      # Skalierungsfaktor (optional einschl. Klammern).
      if ')' in f:
        x = re.sub(r'\(\*(.*)\).*',r'\1',f)
        f = re.sub(r'.*\)',r'',f)
        logging.debug('... factor: "{}"'.format(x))
      else:
        x='1.0'
      facs.append(float(x))  
      #
      #Length
      # Länge des Datenfeldes.
      if '.' in f:
        x = re.sub(r'(.*)\..*',r'\1',f)
        f = re.sub(r'.*\.',r'',f)
      else:  
        x = re.sub(r'([0-9]*).*',r'\1',f)
        f = re.sub(r'([0-9]*)',r'',f)
      x = int(float(x))
      logging.debug('... length: "{}"'.format(x))
      lens.append(x)  
      #
      #Precision
      # Anzahl der Nachkommastellen (bei float-Zahlen).
      x = re.sub(r'^([0-9]*).*',r'\1',f)
      f = re.sub(r'^([0-9]*)',r'',f)
      if x != '':
        x = int(float(x))
        logging.debug('... precis: "{}"'.format(x))
      else:
        x=None
      prec.append(x)
      #
      #Specifier
      # Umwandlungsangabe.
      #Folgende Umwandlungsangaben sind möglich:
      #Spec. Typ        Bytes Beschreibung
      # c    character  1     einzelne Buchstaben
      # d    integer    4     Dezimalzahl
      # hd   integer    2     Dezimalzahl
      # x    integer    4     Hexadezimalzahl
      # hx   integer    2     Hexadezimalzahl
      # f    float      4     Festkommazahl (ohne Exponent)
      # lf   float      8     Festkommazahl (ohne Exponent)
      # e    float      4     Gleitkommazahl (mit Exponent)
      # le   float      8     Gleitkommazahl (mit Exponent)
      # t    integer    4     Binär:Zeitangabe (ohne Datum): vergangene Sekunden
      #                       Text: dd.hh:mm:ss oder hh:mm:ss
      # lt   float      8     Binär: Zeitangabe mit Datum: 
      #                         Vorkommastellen: Anzahl der Tage seit 
      #                           1899-12-30.00:00:00 plus 106
      #                         Nachkommastellen: vergangene Sekunden an diesem Tag
      #                       Text: yyyy-mm-dd.hh:mm:ss
      if f in ['c','d','hd','x','hx','f','lf','e','le','t','lt',]:
        logging.debug('... specif: "{}"'.format(f))
        specs.append(f)
      else:
        raise IOError('unknown format spefifier {}'.format(f))
    
    return(nams,facs,lens,prec,specs)

  # ----------------------------------------------------------------------
  #
  # determine if data are stored externally
  #
  def _get_datfile(self):
    '''
    parses the header dictionary and gets number and kind of dimensions
    '''
    # ascii or binary ?
    mode = self._attrib('mode','text')
    logging.debug('mode: {}'.format(mode))
    # compression strentgth ?
    cmpr = int(self._attrib('cmpr','0'))
    logging.debug('cmpr: {}'.format(cmpr))
    #
    # name of separate datafile (if any)
    datfile = self._attrib('data',None)
    if datfile is None:
      if mode == 'text' and cmpr > 0:
        datfile=re.sub(r'.dmna$','.dmnt.gz',self.file)
      elif mode == 'text' and cmpr > 0:
        datfile=self.file
      elif mode == 'binary' and cmpr == 0:
        datfile=re.sub(r'.dmna$','.dmnb',self.file)
      elif mode == 'binary' and cmpr > 0:
        datfile=re.sub(r'.dmna$','.dmnb.gz',self.file)
    logging.debug('datfile: {}'.format(datfile))
    if cmpr > 0:
      gz=True
    else:
      gz=False
    return(datfile,gz)

  # ----------------------------------------------------------------------
  #
  # read variable definitions
  #
  def _get_data(self):
    '''
    parses the header dictionary and gets number and kind of dimensions
    '''
    dims = self._attrib('dims')
    #
    # get index oder and orientation
    #
    # index sequence gives order (slowest counting to fastest counting)
    # of numbers in file e.g. "k+,j-,i+"
    # index position is position of axis in list seq
    # e.g. x-axis boundaries are in first column in lowb/highb 
    #      x-index "i" is found in last position, direction is + 
    #             -> fastest counting, increasing 
    #             -> along data rows, lowes x left highest x right
    #             
    sequ = self._attrib('sequ').split(',')
    logging.debug('sequ: {}'.format(sequ))
    if len(sequ) != dims :
      print(sequ,len(sequ),dims,len(sequ) - dims)
      raise IOError('number of indices does not match number of dimensions in: '.format(self.file))
    if dims in [1,2,3,4,5]:
      # index names
      inam=['i','j','k','l','m']
      # direction of each index in sequence
      # take scond character of sequence entry,
      # assume "+" if 2nd character is missing
      seqind=[ x[0] for x in sequ[0:dims] ]
      seqdir=[ x[1] if len(x)>1 else '+' for x in sequ[0:dims]]
      # position of each index in sequence
      ipos=[0]*dims
      idir=['']*dims
      for i in range(dims):
        if inam[i] in seqind:
          ipos[i]=seqind.index(inam[i])
          idir[i]=seqdir[ipos[i]]
    else:
      raise IOError('{} dimensions are not supported by this version'.format(dims))
    #
    # index boundaries
    #
    if not ( 'hghb' in self.header.keys() and 'lowb' in self.header.keys() ):
      raise IOError('file does not contain information on grid size: '.format(self.file))
    lowb=[int(x) for x in self.header['lowb'].split()]
    hghb=[int(x) for x in self.header['hghb'].split()]
    ilen=[ x-y+1 for x,y in zip(hghb,lowb)]

    logging.debug('ipos:   {}'.format(ipos))
    logging.debug('idir:   {}'.format(idir))
    logging.debug('ilen:   {}'.format(ilen))

    #
    # how many values per data record
    #
    form = self._attrib('form',None)
    if form is not None:
      (valnams, valfacs, vallens, valprec, valspec) = self._parse_form(form)
      nval=len(valspec)
    else:
      nval=1

    logging.debug('nval:   {}'.format(nval))
    logging.debug('valnams : {}'.format(valnams))
    logging.debug('valfacs : {}'.format(valfacs))
    logging.debug('vallens : {}'.format(vallens))
    logging.debug('valprec : {}'.format(valprec))
    logging.debug('valspecc: {}'.format(valspec))
    
    #
    # ascii or binary ?
    mode = self._attrib('mode','text')
    logging.debug('mode: {}'.format(mode))
    #
    # number format ?
    locl = self._attrib('locl','C')
    logging.debug('locl: {}'.format(mode))
    #
    # select file opening function according to compression
    if self.compressed == True:
      ofct = gzip.open
    else:
      ofct = open
    
    #
    # read the data 
    #
    # number of number-records to read:
    numrec=1
    for i in range(dims):
      numrec=numrec*ilen[i]
    #
    # read all numbers as one big sequence
    numbers=[]
    if mode == 'text':
      # load data from separate data fiel into text buffer
      if self.datfile != self.file:
        with ofct(self.file,'r') as f:
          for x in f.readlines():
            self.text.append( str(x).rstrip('\n') )
      # read starting after header plus '*' line:
      for i,l in enumerate( self.text[ self.header['lines']+1: ] ):
        if '*' in l:
          logging.debug('stopped reading at line {} ("{}")'.format(i,l))
          break
        elif l.strip() != '' :
          for f in l.strip().split():
            spec = valspec[len(numbers)%len(valspec)]
            if spec in ['c']:
              x = f
            elif spec in ['d','hd','x','hx']:
              x = int(_locl_float(f,locl))
            elif spec in ['f','lf','e','le']:
              x = _locl_float(f,locl)
            elif spec in ['t']:
              # dd.hh:mm:ss oder hh:mm:ss
              if '.' in f:
                x = np.timedelta64(int(f.split('.')[0]), 'D')
              else:
                x = np.timedelta64(0, 's')
              x = x + (np.datetime64( '2000-01-01 '+f ) - 
                       np.datetime64( '2000-01-01 00:00:00' ))
            elif spec in ['lt']:
              # yyyy-mm-dd.hh:mm:ss
              x = np.datetime64( f.replace('.',' ') )
            else:
              raise RuntimeError('internal: illegal format specifier: {}'.format(spec))
            numbers.append(x)

    elif mode == 'binary':
      # assemple binary format
      bint={'c':'c','d':'i','hd':'h','x':'i','hx':'h','f':'f','lf':'d','e':'f','le':'d','t':'i','lt':'f',}
      binl={'c':1,'d':4,'hd':2,'x':4,'hx':2,'f':4,'lf':8,'e':4,'le':8,'t':4,'lt':8,}
      # numberformat to read: '<'=little endian 'f'=float
      bf = '<'
      bl = 0
      for i in range(nval):
        bf = bf + bint[valspec[i]]
        bl = bl + binl[valspec[i]]
      # read binary data into list
      with ofct(self.datfile, "rb") as ff:
        for i in range(numrec):
          numbers+=list(struct.unpack(bf, ff.read(bl)))
    else:
      raise IOError('unsopported mode: {}'.format(mode))
      
    logging.debug('numrec : {}'.format(numrec))
    logging.debug('#values: {}'.format(numrec*nval))
    logging.debug('#read  : {}'.format(len(numbers)))

    # split variables to indivitual fields:
    # 123123123123 -> [1111],[2222],[3333]
    values = []
    for i in range(nval):
      # select all values of variable #i
      vn = np.array( [ numbers[i+x*nval] for x in range(numrec) ] )
      # data in the file are in FORTRAN order i.e. last index is counting fastest
      vr = np.reshape(vn,newshape=[ilen[x] for x in ipos],order='C')
      # reorder axes according to "sequ" parameter
      values.append(np.transpose(vr,axes=ipos))
      del(vn,vr)
    #
    # reverse order of values if an index was counting backwards
    #
    for i,v in enumerate(values):
      for k,d in enumerate(idir): 
        if d == '-':
          values[i] = np.flip(values[i],k)
    #
    # make output
    #
    if dims==1:
      out=pd.DataFrame({k:v for k,v in zip(valnams,values)})
      #
      # if timeseries: find time column and convert to POSIXct
      #
      if self.header['artp']=='ZA' and 'te' in out.columns:
        out.loc[:,'te'] = pd.to_datetime(out['te'])
        out.set_index(out['te'])
    else:
      out={k:v for k,v in zip(valnams,values)}
    return (dims,nval,ilen,out)
  # ----------------------------------------------------------------------
  #
  # read file into memory
  #
  def load(self,file,text=False):
    '''
    loads the contents of a dmna file into the object
    
    :param file: filename (optionally including path). \
      If missing, an emtpy
    :param text: (optional) If ``True`` the raw file contents \
      are containted as atrribute `text` in the object. If ``False`` \
      or missing, the raw file contents are discarded after parsing.
    '''
    with open(self.file,'r') as f:
      self.text = [str(x).rstrip('\n') for x in f.readlines()]
    self.header=self._get_header()  
    self.datfile,self.compressed = self._get_datfile()
    self.dims,self.vars,self.shape,self.data=self._get_data()
    if not text == True:
      del self.text
    self.file = file
  # ----------------------------------------------------------------------
  #
  # constructor
  #
  def __init__(self,file=None, var=1, text=False):
    object.__init__(self)
    self.file = file
    if file is not None:
      self.load(file,text)
  # ----------------------------------------------------------------------
  #
  # calculate x/y/z axes values in model coordinates
  #
  def axes(self,ax=None):
    if self.file is None:
      raise AttributeError('no file loaded')
    #
    # "empty" values
    #
    xx = yy = zz = [0.]
    #
    # get axis start and length
    #
    dims = self.dims
    if dims >= 1:
      xlen = self.shape[0]
      xmin = self._attrib('xmin')
    if dims >= 2:
      ylen = self.shape[1]
      ymin = self._attrib('ymin')
    if dims >= 3:
      zlen = self.shape[2]
      sk = self._attrib('sk',None)
    #
    # get spacing
    delta = self._attrib('delta')
    #
    # calculate values
    xx = [ xmin+delta*i for i in range(xlen) ]
    yy = [ ymin+delta*i for i in range(ylen) ]
    if sk is not None:
      zz = [ float(x) for x in sk ]
    else:
      if zlen == 1:
        zz = [0.]
      else:
        raise IOError ('file does not contain level heights: {}'.format(self.file))
    #
    # make dict and return it completely or just one dimension
    axs={'x':xx, 'y':yy, 'z':zz}
    if ax is None:
      return( axs )
    elif ax in ['x', 'y', 'z']:
      return(axs[ax])
    else:
      raise ValueError('unknown axis: {}'.format(ax))
  # ----------------------------------------------------------------------
  #
  # calculate  in Gauss-Krueger coordinates
  #
  def grid(self):
    '''
    calculate grid definition needed for georeferencing
    :returns xlen: number of cells along x-axis
    :returns ylen: number of cells along x-axis
    :returns xll: right-ward position of lower left (southwest) corner
    :returns yll: u-ward position of lower left (southwest) corner
    :returns delta: grid spacing
    '''
    if self.file is None:
      raise AttributeError('no file loaded')
    #
    # get axis start and length
    dims = self.dims
    if dims < 2:
      raise ValueError('file must contain at least two dimensions')
    xlen,ylen = self.shape[0:2]
    xmin = self._attrib('xmin')
    ymin = self._attrib('ymin')
    delta = self._attrib('delta')
    #
    # reference position
    refx = self._attrib('refx',None)
    refy = self._attrib('refy',None)
    if refx is None or refy is None:
      raise ValueError('file does not contain all information on grid')
    #
    # calculate values
    xll=refx+xmin
    yll=refy+ymin
    #
    # return dict
    out = {'xlen': xlen, 'ylen': ylen,
           'xll': xll, 'yll': yll, 'delta':delta }
    return(out)
   
      
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from matplotlib import cm
  logging.basicConfig(level=logging.DEBUG)
  #
  # test axes
  qq=DataFile('../tests/so2-y00a.dmna')
  xy=qq.axes()



  #
  # test 2D
#  dmna=DataFile('../tests/so2-y00a.dmna')
#  blah=dmna.data['con']
#  print(np.shape(blah))
#  print(np.nanmin(blah),np.nanmax(blah))
#  blah[5,10:15,:]=0.
#  plt.contourf(
#               np.transpose(blah[:,:,0]),
#               cmap=cm.get_cmap('YlGnBu')
#               )
  
#  # test 3D
#  dmna=DataFile('../tests/w1018a00.dmna')
#  blah=np.sqrt( dmna.data['Vx']**2 + dmna.data['Vy']**2 )
#  print(np.shape(blah))
#  print(np.nanmin(blah),np.nanmax(blah))
#  blah[5,5:10,:]=0.
#  plt.contourf(
#               np.transpose(blah[:,:,4]),
#               cmap=cm.get_cmap('magma')
#               )
#  
  
#  # test zeitreihe
#  dmna=DataFile('../tests/zeitreihe.dmna')
#  t=dmna.data['te']
#  blah=dmna.data['ua']
#  print(np.shape(blah))
#  print(np.nanmin(blah),np.nanmax(blah))
#  plt.plot(t,blah)