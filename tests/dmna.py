#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:03:56 2019

@author: druee
"""

import unittest

import numpy as np 

import readmet 

class Test_2D(unittest.TestCase):
  def __init__(self,*args):
    unittest.TestCase.__init__(self,*args)
    self.dmna=readmet.dmna.Dmna('tests/so2-y00a.dmna')
  def test_vars(self):
    res = list(self.dmna.data.keys())
    self.assertEqual(res, ['con'])
  def test_shape(self):
    res = np.shape(self.dmna.data['con'])
    self.assertEqual(res, (100,100,1))
  def test_svalues(self):
    res = list(self.dmna.data['con'][80:91,90,0])
    cmp = [ 1.3,  1.3,  1.3,  1.3,  1.4,  1.3,  1.4,  1.3,  1.3,  1.4,  1.3]
    self.assertEqual(res, cmp)
#
##  # test 3D
##  dmna=Dmna('../tests/w1018a00.dmna')
#blah=np.sqrt( dmna.data['Vx']**2 + dmna.data['Vy']**2 )
#print(np.shape(blah))
#print(np.nanmin(blah),np.nanmax(blah))
#blah[5,5:10,:]=0.
#plt.contourf(
#             np.transpose(blah[:,:,4]),
#             cmap=cm.get_cmap('magma')
#             )
#
#
class Test_zeitreihe(unittest.TestCase):
  def __init__(self,*args):
    unittest.TestCase.__init__(self,*args)
    self.dmna=readmet.dmna.Dmna('tests/zeitreihe.dmna')
  def test_vars(self):
    res = list(self.dmna.data.keys())
    self.assertEqual(set(res), set(['te','ra','ua','lm']))
  def test_shape(self):
    res = np.shape(self.dmna.data['ra'])
    self.assertEqual(res, (8784,))
  def test_svalues(self):
    res = list(self.dmna.data['ra'][80:91])
    cmp = [72.0, 97.0, 173.0, 196.0, 215.0, 226.0, 236.0, 242.0, 239.0, 240.0, 241.0]
    self.assertEqual(res, cmp)
