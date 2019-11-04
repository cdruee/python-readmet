#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:43:22 2019

@author: druee
"""

import re
import struct
import logging
import numpy as np
import pandas as pd
#
#
#
class Dmna(object):
  #
  # read header
  #
  def _get_header(self):
    #
    # read the file as text lines and find the divider line "*"
    #
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
    
  def _get_vars(self):
    #
    # get number and kind of dimensions
    #
    if not 'dims' in self.header.keys():
      raise IOError('file does not contain number of dimensions: {}'.format(self.file))
    dims=int(self.header['dims'])
    if dims > 3 :
      raise RuntimeError('this function only supports up to three dimensions')
    else:
      logging.info('dims: {}'.format(dims))
    if not 'artp' in self.header.keys():
      artp=""
    else:
      artp=self.header['artp']
    logging.debug('artp: {}'.format(artp))
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
    if not 'sequ' in self.header.keys():
      raise IOError('file does not contain information on index order: '.format(self.file))
    sequ=self.header['sequ'].split(',')
    logging.debug('sequ: {}'.format(sequ))
    if len(sequ) != dims :
      raise IOError('number of indices does not match number of dimensions in: '.format(self.file))
    if dims in [1,2,3]:
      # index names
      inam=['i','j','k']
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
    seqlen=[None]*dims
    for i in range(dims):
      seqlen[int(ipos[i])]=ilen[i]

    logging.debug('ipos:   {}'.format(ipos))
    logging.debug('idir:   {}'.format(idir))
    logging.debug('ilen:   {}'.format(ilen))
    logging.debug('seqdir: {}'.format(seqdir))
    logging.debug('seqlen: {}'.format(seqlen))

    self.dims=dims
    self.ipos=ipos
    self.idir=idir
    self.ilen=ilen
    self.seqdir=seqdir
    self.seqlen=seqlen
    return(0)

  def _get_data(self):
    #
    # read the actual data from file
    #
    #
    # ascii or binary ?
    if 'mode' in self.header.keys():
      mode=self.header['mode']
    else:
      mode='text'
    logging.debug('mode: {}'.format(mode))
    #
    # how many values per data record
    #
    if 'form' in self.header.keys():
      forms=self.header['form'].split(' ')
      nval=len(forms)
      valnams=[ x.split('%')[0] for x in forms ]
    else:
      nval=1
    logging.debug('nval:   {}'.format(nval))
    #
    # read the data 
    #
    # number of number records to read:
    numrec=1
    for i in range(self.dims):
      numrec=numrec*self.ilen[i]
    if mode == 'text':
      if self.dims > 1 and nval > 1 :
        raise IOError('number of values >1 for dimensions >1 not implemented ')
      #
      # layers are stored as consecutive 2D arrays 
      numbers=[]
      ptr = self.header['lines']+1
      while True:
        try:
          fwd = self.text[ptr:].index('')
        except ValueError:
          break
        for l in self.text[ptr:ptr+fwd]:
          for f in l.split():
            numbers.append(float(f))
        ptr=ptr+fwd+2
    elif mode == 'binary':
      binfile=re.sub(r'.dmna$','.dmnb',self.file)
      # numberformat to read:
      # mans '<'=little endian 'f'=float
      fm = '<'+'f'*nval
      # read binary data into list
      numbers=[]
      with open(binfile, "rb") as ff:
        for i in range(numrec):
          numbers+=list(struct.unpack(fm, ff.read(nval*4)))
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
      va = np.array( [ numbers[i+x*nval] for x in range(numrec) ] )
      if self.ipos==[2,1,0] :
        # data in the file are in FORTRAN order i.e. last index is counting fastest
        values.append( np.reshape(va,newshape=self.ilen,order='F')) 
      elif self.ipos==[0,1,2] :
        # data in the file are in C order i.e. first index is counting fastest
        values.append( np.reshape(va,newshape=self.ilen,order='C'))
      else:
        raise RuntimeError('this order of indices is not yet implemented')
    #
    # reverse order of values if an index was counting backwards
    #
    for i,v in enumerate(values):
#      for k,d in enumerate(reversed(self.idir)): 
      for k,d in enumerate(self.idir): 
        print('idir: '+d)
        if d == '-':
          print('flip {} dir {}'.format(valnams[i],k))
          values[i] = np.flip(values[i],k)
    #
    # make output
    #
    if self.dims==1:
      out=pd.DataFrame({k:v for k,v in zip(valnams,values)})
      #
      # if timeseries: find time column and convert to POSIXct
      #
      if self.artp=='ZA' and 'te' in out.columns:
        out.loc[:,'te'] = pd.to_datetime(out['te'])
        out.set_index(out['te'])
    else:
      out={k:v for k,v in zip(valnams,values)}
    return (out)
  #
  # read file into memory
  #
  def load(self,file=None,text=False):
    with open(self.file,'r') as f:
      self.text = [str(x).rstrip('\n') for x in f.readlines()]
#      self.text = [str(x) for x in f.readlines()]
    self.header=self._get_header()  
    self.vars=self._get_vars()
    self.data=self._get_data()
    if not text == True:
      del self.text
  #
  # constructor
  #
  def __init__(self,file=None, var=1):
    object.__init__(self)
    self.file = file
    if file is not None:
      self.load(file)
## ------------------------------------------------------------------------
##
## define a helper function following a discussion at
## http://r.789695.n4.nabble.com/distributing-the-values-of-data-frame-to-a-vector-based-on-td837896.html
##
#revdim <- function(arr, di=NA) {
#  if ( is.vector(arr) ){
#    return(rev(arr))
#  } else {
#    ndim = dim(arr)
#    dims = length(ndim)
#    idim = 1:dims
#    rdim = idim %in% di 
#    if ( sum(rdim) > 0 ) {
#      rind <- function(i){
#        if (! i %in% idim) {
#          return(0)
#        } else if (rdim[i]) {
#          return(rev(1:ndim[i]))
#        } else {
#          return(1:ndim[i])
#        }
#      }
#      do.call("[", c(list(arr), sapply(idim, rind, simplify=FALSE),drop=FALSE))
#    } else {
#      return(arr)
#    }
#  }
#}

#dmna.axes <- function(file) {
#  header <- dmna.header(file)
#  #
#  # get axes length
#  #
#  if ( ! ( "dims" %in% names(header))) {
#    stop (paste(file,"does not contain number of dimensions"))
#  }
#  dims=header$dims
#  if ( ! ( "axes" %in% names(header))) {
#   if ( dims >= 1 & dims <= 3 ) {
#     header$axes='xyz'[1:dims]
#   }else{
#     stop (paste(file,"does not contain names of axes"))
#   } 
#  }
#  if ( ! ( "hghb" %in% names(header) &&  "lowb" %in% names(header))) {
#    stop (paste(file,"does not contain index value ranges"))
#  }
#  lowb=read.table(header=F,text=header$lowb)
#  hghb=read.table(header=F,text=header$hghb)
#
#  xlen=hghb[,1]-lowb[,1]+1
#  if (dims>1) {
#    ylen=hghb[,2]-lowb[,2]+1
#  } else {
#    ylen=1
#  }
#  if (dims>2) {
#    zlen=hghb[,3]-lowb[,3]+1
#  } else {
#    zlen=1
#  }
#  #
#  # get axes start
#  #
#  if ( ! (  "xmin" %in% names(header) 
#         && "ymin" %in% names(header) 
#         && "delta" %in% names(header))) {
#    stop (paste(file,"does not contain all information on axes"))
#  }
#  xmin=as.numeric(header$xmin)
#  ymin=as.numeric(header$ymin)
#  delta=as.numeric(header$delta)
#  x=xmin+delta*((1:xlen)-1)
#  if (dims>1) {
#    y=ymin+delta*((1:ylen)-1)
#  } else {
#    y=0:0
#  }
#  if (zlen==1) {
#    return(list(x=x,y=y))
#  } else {
#    if ( ! (  "sk" %in% names(header))) {
#      stop (paste(file,"does not contain level heights"))
#    }
#    sk=as.matrix(read.table(header=F,text=header$sk))
#    if (! ( hghb[,3] <= length(sk) && lowb[,3] >= 1 ) ) {
#      stop (paste(file,"level indices outside givel level heights"))
#    }
#    z=sk[lowb[,3]:hghb[,3]]
#    return(list(x=x,y=y,z=z))
#  }
#}
#
#
#dmna.grid <- function(file) {
#  header <- dmna.header(file)
#  #
#  # get axes length
#  #
#  if ( ! ( "dims" %in% names(header))) {
#    stop (paste(file,"does not contain number of dimensions"))
#  }
#  dims=header$dims
#  if ( header$artp == "ZA" | dims<2 ) {
#    stop (paste("this function does not apply to timeseries"))
#  }
#  if ( ! ( "hghb" %in% names(header) 
#         &&  "lowb" %in% names(header))) {
#    stop (paste(file,"does not contain index value ranges"))
#  }
#  lowb=read.table(header=F,text=header$lowb)
#  hghb=read.table(header=F,text=header$hghb)
#  xlen=hghb[,1]-lowb[,1]+1
#  ylen=hghb[,2]-lowb[,2]+1
#  #
#  # get axes start
#  #
#  if ( ! (  "xmin" %in% names(header) 
#         && "ymin" %in% names(header) 
#         && "delta" %in% names(header))) {
#    stop (paste(file,"does not contain all information on axes"))
#  }
#  xmin=as.numeric(header$xmin)
#  ymin=as.numeric(header$ymin)
#  delta=as.numeric(header$delta)
#  #
#  # reference position
#  #
#  if ( ! (  "refx" %in% names(header) 
#            && "refy" %in% names(header))) {
#    stop (paste(file,"does not contain all information on axes"))
#  }
#  refx=as.numeric(header$refx)
#  refy=as.numeric(header$refy)
#  #
#  # lower left corner reference:
#  #
#  xll=refx+xmin
#  yll=refy+ymin
#  #
#  # return list
#  #
#  out=c(xlen,ylen,xll,yll,delta)
#  names(out)=c("xlen","ylen","xll","yll","delta")
#  return(out)
#}
#
#.Random.seed <-
#c(403L, 10L, -339291239L, 1063529515L, 545478938L, -1965270052L, 
#-868639265L, -1535243851L, -844576948L, -338379358L, 538193149L, 
#826814967L, 2100707102L, 20324824L, 2058246971L, 1292086489L, 
#1598578792L, 795166102L, -310035183L, -1930129725L, -357478702L, 
#-189507740L, 620399271L, -720677891L, 322659828L, -128671014L, 
#-2046946267L, -128345793L, 431824582L, 445248336L, 148441971L, 
#937237905L, 2100494272L, 1036442014L, 1670693737L, 1877972891L, 
#1863663274L, -213215028L, -860821105L, 446012869L, 63816060L, 
#592007634L, -479662771L, 1922715655L, -1509867538L, -1096579384L, 
#1940322763L, 1139797033L, -328942984L, 268347302L, 1663316417L, 
#-1325636013L, 2096901378L, -1045570508L, 665648567L, 897487277L, 
#-846941500L, -790878486L, -969864875L, 2033753839L, 979229942L, 
#-447624160L, 1358515683L, 261729921L, -1547532048L, -2146498674L, 
#-2095597319L, -1229839733L, -1249369414L, 1622183356L, 493795071L, 
#-1489159595L, -1979089556L, 766965186L, 944584221L, -1942212969L, 
#1919166462L, -1538346056L, 567619483L, 595588601L, -1210944184L, 
#-1527644042L, 1901063601L, -1297624093L, -1882746702L, 1860271364L, 
#1958846023L, 1242655005L, 1874552724L, 1610935098L, 26310533L, 
#-686742433L, -402673946L, 960910704L, 1874435731L, 1879893809L, 
#1123442144L, -1252128130L, 251959561L, 395000635L, -490630390L, 
#917617388L, 258815855L, -1039370075L, -1038912932L, -1918139854L, 
#-290543315L, 2109353959L, 1111414542L, 1117325544L, 306584299L, 
#432223113L, -1863954664L, -1865829818L, -1587580255L, 771814707L, 
#-2047079262L, -1573526636L, -2107442665L, 1126596109L, 1971706660L, 
#-2008135926L, 366987893L, -128468529L, 1298596566L, 1367762560L, 
#1105488067L, -1756707231L, -2096161072L, -707360722L, -1700184615L, 
#-12164629L, -476912550L, -1963368932L, 1574530847L, 256525173L, 
#1892936204L, -274371614L, -559376195L, 231212855L, 1493815262L, 
#-1477411560L, 1564790907L, -1457437159L, -1511231832L, -20243370L, 
#647084241L, -1616419709L, -2017786094L, -2107628892L, 900001383L, 
#-1041735747L, 1997558068L, 1487393434L, 1114280165L, -940207489L, 
#-472491002L, 1281699728L, -1505078733L, 1041320401L, 276044672L, 
#-1761683874L, 467963305L, -695264165L, -1253997462L, 547028236L, 
#1878499151L, 1704188933L, -1432567108L, 87888786L, 1451839629L, 
#653125319L, 1797255086L, -1532356088L, 1866882571L, 934106857L, 
#1646834104L, -1323250074L, 1933511425L, -711762541L, 1436513858L, 
#-223565324L, 469603319L, -800528403L, -1860009852L, 443700906L, 
#-163902443L, -874922833L, -1441566026L, 1718235360L, 1464957987L, 
#1016562497L, 1816333872L, 1719053390L, 1080485049L, -997891893L, 
#662048122L, -1499049860L, -1339493185L, -382828139L, -54883924L, 
#-189686910L, -976772771L, 384147543L, -2027604162L, -1720828040L, 
#773002331L, 1849402297L, -74227192L, 289966262L, 1249259505L, 
#-789079517L, 488182386L, -1587935036L, 1908153991L, 207738461L, 
#-2093451180L, 401836666L, -182881851L, -650435809L, -1579544922L, 
#-1392140240L, 897942227L, 1247885553L, -474594272L, 1907704510L, 
#1194534601L, 251454587L, -95036854L, 1171764156L, 1059568946L, 
#-870735936L, -963636684L, 1468980168L, -1904786246L, 1003163536L, 
#-838961684L, -39796652L, 1186381458L, -1212450016L, 631254476L, 
#-1786881056L, 1366950594L, -282977752L, 476094124L, 931263164L, 
#-568733198L, 648169776L, -1308029788L, -486997320L, -208278694L, 
#-1089303152L, -307908228L, 1061457172L, -1276991166L, -1771273472L, 
#449125308L, -1618527408L, 387154210L, -2078155448L, 320665612L, 
#-1016121252L, 1635165394L, -578009088L, -502269100L, 1804542440L, 
#114300922L, -1355376944L, 500032812L, 940804340L, 526698386L, 
#1697610848L, -1602290420L, -1731770656L, 1074421090L, 257344168L, 
#716495436L, -1730594948L, -543888590L, -1229144464L, -1295036L, 
#-1442808872L, 562679930L, 1319282224L, -385414468L, 1025599316L, 
#153279042L, -1134056544L, -234583876L, -2119294512L, 544327010L, 
#1823593448L, -135039028L, 1292121852L, -680603662L, 1878187136L, 
#2143344884L, -218696568L, -433035974L, 2013053776L, 721357932L, 
#-1327677292L, -1820092078L, -875774240L, -932220724L, -1447561056L, 
#1646246274L, 614512872L, -49841940L, -1861470148L, -395313038L, 
#1592584304L, 1930379812L, -36437448L, 89131546L, -1697025840L, 
#1132623420L, -1118647148L, -2012130174L, 923109312L, 441855292L, 
#-1809244528L, 1866085410L, -1051783160L, -1832575860L, -129925092L, 
#2014413970L, 26815616L, 1686676500L, -1959927000L, -962546630L, 
#1249455312L, -317909908L, 1187067572L, 1106755602L, -1262747680L, 
#534244300L, -1751769632L, 1266043938L, 999494952L, -1367639796L, 
#748572220L, 63423730L, 1799231728L, -1105488060L, 983060568L, 
#401564218L, -1690564368L, -1236196676L, -237778412L, -1305604926L, 
#-757393824L, 1055712636L, -1139301168L, -1999694814L, 1176397992L, 
#-1011573364L, 1055326652L, -1944532942L, 568152896L, -235757900L, 
#-452456120L, 908432058L, -1747263728L, -278667028L, 951793748L, 
#1971144466L, -578141792L, -559896628L, -538730656L, -225440574L, 
#1145425448L, -370841172L, -1649073348L, -647642254L, 720047920L, 
#-418471900L, -197960136L, -1959644454L, -607766640L, 1383486716L, 
#-1889276524L, -311983294L, 701106688L, 841631804L, 1791435344L, 
#967844002L, -244063416L, -469517812L, -1657939236L, -130305966L, 
#322571392L, 30328660L, -2025971736L, 842117882L, 673228624L, 
#921224236L, -93222540L, 921456274L, -1121828256L, -642142068L, 
#-459091616L, 296941794L, -1440748376L, -1942702388L, -2045183492L, 
#-540709966L, -780286992L, -1363754556L, -2106917544L, -1400262278L, 
#115683760L, 2059053500L, 1540218452L, -1232324158L, 1236575008L, 
#1640043452L, -1234489264L, 683809762L, -861585560L, 1685043020L, 
#-703205636L, -307386638L, -208790272L, -1529291660L, 291163656L, 
#-1799477830L, 1827333584L, -1464735764L, 1944857492L, 1440385234L, 
#688567392L, -1024627380L, -1299000544L, 1341590146L, 1009603048L, 
#1901616108L, -923775428L, -1275443598L, -427086992L, -104583900L, 
#-1933305288L, 396886426L, 40709584L, 2059552188L, 606580372L, 
#-1611976062L, 804433344L, -1804855492L, -1569586544L, 1645176610L, 
#1638761992L, 802448140L, -1173646436L, 1052384274L, 411446144L, 
#337708548L, -1179725387L, -476057953L, 1280392952L, 654090966L, 
#1989179235L, 582143781L, 17515714L, -1070919248L, -1422022183L, 
#-750827701L, -1916252740L, -1867391510L, -1082889697L, -1369997143L, 
#-1142003650L, 518277420L, 1268573053L, 429767111L, 1030110384L, 
#-1848706354L, 964210683L, -1373685667L, 1689255818L, 137605032L, 
#1930183953L, -623166461L, -2129152252L, -418989262L, -1652426361L, 
#-742744111L, -2040835754L, -95973580L, -1238551995L, -1975279985L, 
#1323587592L, 702082086L, -1136007981L, -1333008843L, -1097769582L, 
#2047517024L, -658851639L, 255663099L, 1055693772L, -1858349478L, 
#-1169048753L, 323990041L, -931835858L, -905699780L, 2045645741L, 
#-757916585L, 1545820576L, -435748098L, -993422901L, 1656281037L, 
#721132282L, 1721631032L, -959517983L, -1818616429L, 764560820L, 
#-695724350L, -2013169961L, 1409316641L, 2123624166L, 1518643172L, 
#1678513941L, 692719423L, -28759336L, -1070952714L, 410822915L, 
#1592457797L, -214861214L, -1899357744L, 1274398457L, -544238549L, 
#1554469788L, -135001142L, -668515137L, 187033097L, 376483614L, 
#-562276660L, 1893971613L, 1883767591L, -815977136L, -619162642L, 
#-540812133L, -1260701955L, -519717398L, -1954787704L, 1357674545L, 
#269075107L, -1428026716L, -2100143278L, -642698585L, -1302786319L, 
#747626038L, -97795372L, -1235577243L, 1930708079L, -1358938456L, 
#-167666682L, 1642932531L, 1156134293L, 1744050674L, 2058573376L, 
#-1886947415L, -564561381L, -1729014804L, -574114822L, -2061025233L, 
#1171681337L, 1068958286L, 1662370460L, -1890793459L, 1276499063L, 
#82805888L, 1915409118L, -2063695445L, -1444070739L, 740859674L, 
#-43794472L, 1043532865L, -25920781L, -1166787308L, -1766127454L, 
#1287538103L, 709548417L, 658890630L, 1001835972L, 1665838325L, 
#1447347551L, 1078060984L, -1803826538L, -1193118557L, -118632859L, 
#1511766914L, 112031984L, -171188071L, -969525109L, 110567804L, 
#-198025302L, 353574879L, 1419791721L, 39367038L, -466640532L, 
#1002772925L, 398150279L, 1284443760L, -1193978610L, 1606695483L, 
#-2024472291L, -649448502L, -1806521112L, 2083491793L, 142156227L, 
#-1437136188L, 807487858L, -911338425L, -1098205167L, 849453462L, 
#-1131481868L, 102301317L, 185817039L, -1764337976L, -220164250L, 
#-366479981L, 1480034037L, -1090509102L, -2086702816L, -637036671L
#)
      
      
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from matplotlib import cm
  logging.basicConfig(level=logging.DEBUG)
  #
  # test 2D
#  dmna=Dmna('../tests/so2-y00a.dmna')
#  blah=dmna.data['con']
#  print(np.shape(blah))
#  print(np.nanmin(blah),np.nanmax(blah))
#  blah[5,10:15,:]=0.
#  plt.contourf(
#               np.transpose(blah[:,:,0]),
#               cmap=cm.get_cmap('YlGnBu')
#               )
  
#  # test 3D
  dmna=Dmna('../tests/w1018a00.dmna')
  blah=np.sqrt( dmna.data['Vx']**2 + dmna.data['Vy']**2 )
  print(np.shape(blah))
  print(np.nanmin(blah),np.nanmax(blah))
  blah[5,5:10,:]=0.
  plt.contourf(
               np.transpose(blah[:,:,2]),
               cmap=cm.get_cmap('magma')
               )
  