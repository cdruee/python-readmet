#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module contains functions and objects for
handling (mostly reading) less popular or vendor-specific
data file formats used in meteorology and neighboring
sciences.
'''

__all__ = ['akterm', 'dmna', 'scintec1', 'toa5']

from . import akterm
from . import dmna
from . import scintec1
from . import toa5

__version__ = '0.6.6'
