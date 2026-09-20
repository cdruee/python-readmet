#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module conatins functions and objects for
handling (mostly reading) less popular or vendor-specific
data file formats used in meterology and neighboring
sciences.
'''

__all__ = ['akterm', 'dmna', 'hpl', 'scintec1', 'toa5',
           '__title__', '__description__', '__version__',
           '__url__', '__author__', '__author_email__',
           '__license__', '__copyright__']

from . import akterm
from . import dmna
from . import hpl
from . import scintec1
from . import toa5

from ._metadata import __title__, __description__, __version__
from ._metadata import __url__, __author__, __author_email__
from ._metadata import __license__, __copyright__
