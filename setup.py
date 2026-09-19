#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Compatibility shim only. All metadata and build configuration live in
# pyproject.toml; this file exists solely so that tooling which still
# looks for setup.py (notably Debian's dh-python/pybuild, invoked via
# dh_make --python in ci_build_deb.sh) auto-detects this as a Python
# package the same way it always has. `setup()` is called with no
# arguments and does not duplicate anything from pyproject.toml.
from setuptools import setup

setup()
