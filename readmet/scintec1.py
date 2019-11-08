#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The classes and functions in this category handle files 
in format "Scintec FORMAT-1" and  "Scintec FORMAT-1.1"
created by Scintec AG, Rottenburg, Germany
(http://scintec.com)

The most comprehensive description of "FORMAT-1" can be found
in the sodar software manual [APRu127]_ and of "FORMAT-1.1".
in the scintillometer software manual [SRun115]_.
'''

import re
import glob
import logging
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------
#
#
#
class DataFile(object):
  '''
  object class that holds data and metadata of a Scintec-1 file
  
  :param file: filename (optionally including path). \
    If missing, an emtpy object is returned
  :param text: (optional) If ``True`` the raw file contents \
    are containted as atrribute `text` in the object. If ``False`` \
    or missing, the raw file contents are discarded after parsing.
  :attrib blah: lorem ipsum 
  '''

  file = None 
  ''' name of file loaded into object '''
  header = None
  ''' dictionary containing the header entries as strings'''
  comments = None
  ''' dictionary containing the comments header entries as strings'''
  vars = None
  ''' ``pandas.Dataframe`` containing information of the variables
      The index contains the variable symbol.
      The columns are "label","symbol","unit","type","error_mask","gap_value"
      for each variable. '''
  nonprofile = None
  ''' ``pandas.Dataframe`` containing the non-profile data 
      (i.e. scalar timeseries) from the file loaded.
      The index is time, each column represents one variable. '''
  profile = None
  ''' dictonary containing the profile data from the file loaded.
      The keys are the variable names. 
      The values are of type ``pandas.DataFrame`` with time as index,
      and the measurement levels as columns. '''
  text = None 
  ''' text contents the file loaded with the (decompressed)
      text contents of an eventual external `datfile` appended '''

  #
  # read header "header"
  #
  def _get_header(self):
    #
    # read the file as text lines and check magic
    #
    header={}
    if self.text[0] == "FORMAT-1":
      header["version"]="1.0"
      header["starttime"]=pd.to_datetime(self.text[1][0:19])
      header["filecount"]=int(self.text[1][19:])
      header["instrument"]=self.text[2].strip()
      header["commentlines"],header["variables"],header["heightlevels"]=[int(x) for x in self.text[3].split()]
      header["fixedlines"]=5
      header["datatype"]="Main Data"
    elif self.text[0] == "FORMAT-1.1":
      header["version"]="1.1"
      header["starttime"]=pd.to_datetime(self.text[1][0:20])
      header["instrument"]=self.text[2].strip()
      header["commentlines"],header["variables"]=[int(x) for x in self.text[3].split()]
      header["fixedlines"]=5
      typeline=header["fixedlines"]+header["commentlines"]
      header["datatype"]=self.text[typeline].strip()
    else:
      raise ValueError('{} is not in Scintec format-1.x format'.format(self.file))
    return(header)
  #
  # read the comments
  #
  def _get_comments(self):
    comments={}
    firstline=self.header['fixedlines']
    lastline=self.header['fixedlines']+self.header['commentlines']
    #
    # ignore commented lines
    pat=re.compile('^\ *#')
    lines=[ x for x in self.text if not pat.match(x) ]
    #
    for pointer in range(firstline,lastline):
      fields = lines[pointer].split(':')
      if len(fields) != 2 :
        raise ValueError('cannot comprehend comment line: {}'.format(lines[pointer]))
      key=fields[0].strip()
      value=fields[1].strip()
      comments[key]=value
    return(comments)  
  #
  # read variable definitions
  #
  def _get_variables(self):
    columns=["label","symbol","unit","type","error_mask","gap_value"]
    vars = pd.DataFrame(columns=columns)
    idx = 0
    #
    # ignore commented lines
    pat=re.compile('^\ *#')
    lines=[ x for x in self.text if not pat.match(x) ]
    #
    # read actual definitions
    #
    firstline=self.header['fixedlines']+self.header['commentlines']+1
    lastline=self.header['fixedlines']+self.header['commentlines']+1+self.header['variables']+1
    for pointer in range(firstline,lastline):
      fields = [x.strip() for x in  lines[pointer].split('#')]
      # fix symbol for error code:
      if fields[0] == 'error code' and fields[1] != 'error':
        fields[2] = fields[1]
        fields[1] = 'error'
        fields[4] = ''
      # omit Time line:
      if fields[0] == "Time" and len(fields) == 5 :
        continue
      vv = pd.DataFrame(dict(zip(columns,fields)), index=[idx])
      vars = pd.concat([vars,vv])
      idx = idx + 1 
    # make shure entries have proper type
    gv=[np.nan] * len(vars['gap_value'])
    for i,x in enumerate(vars['gap_value']):
      try:
        gv[i] = float(x)
      except:
        pass
    vars['gap_value']=gv
    #  
    return(vars)
  #
  # read non-profile data
  #
  def _get_nonprofile(self):
    npdata = None
    #
    # ignore commented lines
    pat=re.compile('^\ *#')
    lines=[ x for x in self.text if not pat.match(x) ]
    #
    # switch Format versions
    #
    if self.header['version']=='1.0':
      #
      # Scintec Format-1
      #
      # if nonprofile variuable were recorded
      #
      if 'NS' in self.vars['type']:
        #
        # loop data
        #
        pointer=self.header['fixedlines']
        while pointer < len(lines):
          # look for date/time
          if re.match('^....-..-.. ',lines[pointer]):
            # get date/time
            field = lines[pointer].split()
            datetime = ' '.join([(field[0],field[1])])
            # get names
            line = re.sub('^\ *#','',lines[pointer+1])
            names=line.split()
            values=[float(x) for x in lines[pointer+2].split()]
            vv={names[i]:v for i,v in enumerate(values)}
            df=pd.DataFrame.from_dict(vv)
            df.set_index(pd.to_datetime(datetime))
            if npdata is None:
              npdata=df
            else:
              npdata = pd.concat(npdata,df)
            pointer=pointer+3
          else:      
            pointer=pointer+1
        return(npdata)
      else:
        return(None)
    elif self.header['version']=='1.1':
      #  
      # Scintec Format-1.1
      #
      pointer=self.header['fixedlines']+1+self.header['commentlines']+1+self.header['variables']+1
      idx=0
      while pointer < len(lines):
        # get date/time
        field = re.split('[\ \t]+',lines[pointer])
        timestr = field[0].split('/')
        datetime = pd.to_datetime(timestr[1],format="%Y-%m-%dT%H:%M:%SZ")
        values = [float(x) if x != 'N/A' else np.nan for x in field[1:] ]
        if len(values) != len(self.vars['symbol']):
          logging.warn('incomplete line #{}'.format(pointer))
        else:
          names=self.vars['symbol']
          vv={names[i]:v for i,v in enumerate(values)}
          df=pd.DataFrame(vv,index=[datetime])
          idx = idx + 1
          if npdata is None:
            npdata=df
          else:
            npdata = pd.concat([npdata,df])
          pointer=pointer+3
      return(npdata)
    else:
      raise RuntimeError('unknown Format version {} reading nonprofile data'.format(self.header['version']))
  #
  # read profile data
  #
  def _get_profile(self):
    #
    # ignore commented lines
    pat=re.compile('^\ *#')
    lines=[ x for x in self.text if not pat.match(x) ]
    #
    # switch Format versions
    #
    if self.header['version'] == '1.0':
      #
      # Scintec Format-1
      #
      #
      if 'NS' in self.vars['type']:
        npvars = True
      else:
        npvars = False
      levels = self.header['heightlevels']
      # initialize
      pointer = 5
      fields={}
      # loop data
      while pointer < len(lines):
        # look for date/time
        if re.match('^....-..-.. ',lines[pointer]):
            # get date/time
            field = lines[pointer].split()
            datetime = ' '.join([field[0],field[1]])
            logging.info('   ... reading {}{}'.format(field[0],field[1]))
            # jump over non-profile data
            if npvars:
              pointer = pointer+3
            else:
              pointer = pointer+1
            # get the lines that form the block of numbers
            #block = '\n'.join(lines[pointer:(pointer+levels)])
            block = lines[pointer:(pointer+levels)]
            # get names of the profile variables
            names=[ x for i,x in enumerate(self.vars['symbol']) if self.vars['type'][i] != 'NS' ]
            # read the numbers
            df=pd.DataFrame([x.split() for x in block],columns=names)
            heights=[ str(x) for x in df['z'].values ]
            df = df.drop('z',1)
            for c in df.columns:
              nf=pd.DataFrame(df[c]).transpose()
              # replace field name by time
              nf.set_index(pd.DatetimeIndex([pd.to_datetime(datetime)]),inplace=True)
              # replace field number by heights
              nf.columns = heights
              if not c in fields.keys():
                # initialize output on first pass
                fields[c] = nf
              else:
                # append read data to output
                fields[c]=pd.concat([fields[c],nf])
              del(nf)
            # jump to end of block
            pointer=pointer+levels-1
        # continue search for next block
        pointer=pointer+1
      # replace gap values in each field by NA
      if fields is not None:
        for c in fields.keys():
          gap = self.vars['gap_value'][self.vars['symbol']==c]
          logging.dbug('gap {}'.format(gap))
          fields[c].replace(gap,np.nan,inplace=True)
    elif self.header["version"]=="1.1":
      #  
      # Scintec Format-1.1
      #
      # (no grid variables)
      fields={}
    else:
      raise ValueError('unknown Format version {} reading nonprofile data'.format(self.header['version']))
      #  
    return(fields)
  #
  # read file into memory
  #
  def load(self,file=None,text=False):
    with open(self.file,'r') as f:
      self.text = [x.rstrip() for x in f.readlines()]
    self.header=self._get_header()  
    self.comments=self._get_comments()  
    self.vars=self._get_variables()
    self.nonprofile=self._get_nonprofile()
    self.profile=self._get_profile()
    if not text == True:
      del self.text
  #
  # constructor
  #
  def __init__(self,file=None,text=None):
    object.__init__(self)
    self.file = file
    if file is not None:
      self.load(file,text)

# ------------------------------------------------------------------------
def read(pattern):
  '''
  read a sequence of Scintec-1 files into one data structue
  
  :param pattern: a `globbing pattern <https://en.wikipedia.org/wiki/Glob_(programming)>`_ \
    describing one or multiple filenames or paths
  :returns: contatined variables as dictionary with the variable names as keys. \
    Each variable is returned as a `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ \
    with date/time as index of type `pandas.DatetimeIndex <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex>`_
  '''
  # expand globbing pattern
  if isinstance(pattern,list):
    files=[]
    for x in pattern:
      # expand globbing pattern
      logging.debug('globbing pattern is: {}'.format(x))
      ex = glob.glob(x)
      logging.debug('expanded file list : {}'.format(ex))
      # append to full list of files:
      files += ex
  else:
    files = glob.glob(pattern)
  logging.debug('list of files to open: {}'.format(files))
  # read all files
  fields=None
  series=None
  for i,file in enumerate(files):
    logging.info('opening file #{}:{}'.format(i,file))
    scintec1 = Scintec1(file)
    if len(scintec1.vars) > 0:
      if fields is None or series is None:
        fields = scintec1.profile
        series = scintec1.nonprofile
      else:
        warn_duplicated = False
        fmore = scintec1.profile
        for c in fmore.keys():
          if c in fields.keys():
            if isinstance(fmore[c],type(fields[c])):
              if any(x in fields.index for x in fmore.index):
                warn_duplicated = True
              pd.concat([fields[c],fmore[c]]).drop_duplicates(keep='last')
            else:
               raise TypeError('dont know how to handle variable {}'.format(c))
          else:
             logging.warn('new variable "{}" in file {}'.format(c,scintec1.file))
             logging.warn('{}'.format(fmore.keys()))
        smore = scintec1.nonprofile
        if any(x in series.index for x in smore.index):
          warn_duplicated = True
        series = pd.concat([series,smore]).drop_duplicates(keep='last')
        #
        if warn_duplicated == True:
          logging.warn('repeated times in file {}')
    del(scintec1)
  # sort data by time
  if len(fields) > 0:
    for c in fields.keys():
      if isinstance(fields[c],pd.DataFrame):
        fields[c].sort_index(inplace=True)
      else:
        raise ValueError('sort time: dont know how to handle field {}'.format(c))
  # sort data by time
  if series is not None and len(series.keys()) > 0:
    # sort data by time
    if isinstance(series,pd.DataFrame):
      series.sort_index(inplace=True)
    else:
      raise ValueError('sort time: dont know how to handle series {}'.format(c))
  # add non-profile data
  for c in series.keys():
    if not ( c == "time" and "time" in fields.keys() ):
      fields[c]=pd.DataFrame(series[c])
  return(fields)

if __name__ == '__main__':
  a=Scintec1('/localdata/druee/projekte/cats/dat/MFAS/data/171101.mnd')
  b=Scintec1('/localdata/druee/projekte/cats/dat/BLS/SPU-111-101/2017-11-01.mnd')
             
  