#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module conatins functions and objects for 
handling (mostly reading) less popular or vendor-specific
data file formats used in meterology and neighboring 
sciences.
'''

import numpy as np
import pandas as pd

from .__version__ import __title__, __description__, __version__
from .__version__ import __url__, __author__, __author_email__
from .__version__ import __license__, __copyright__

from . import scintec1
from . import dmna

