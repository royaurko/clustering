#!/usr/bin/python
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="tcluster",
    version="1.0.0",
    author='Aurko Roy, Sadra Yazdanbod, Daniel Zink',
    author_email="aurko@gatech.edu",
    description=("A threshold clustering algorithm"),
    license="BSD",
    keywords="clustering",
    packages=['tcluster'],
    package_dir={'tcluster': ''},
    long_description=read('Readme.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
