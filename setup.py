from setuptools import setup
#from distutils.core import setup

import readmet as lib

setup(
    name = lib.__title__ ,
    version = lib.__version__ ,
    packages = [lib.__title__] ,
    package_dir={lib.__title__: lib.__title__} ,
    test_suite='tests',
#    license = lib.__license__ ,
    author = lib.__author__ ,
    author_email = lib.__author_email__ ,
    url = lib.__url__ , 
#    long_description=open('README.txt').read(),
)

