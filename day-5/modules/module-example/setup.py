
import sys
import os
from setuptools import setup, find_packages

NAME = "geox_statistics"
VERSION = "0.0.1"

REQUIRES = ["numpy"] # modules we need
HERE = os.path.dirname(__file__)
setup(
    name=NAME,
    version=VERSION,
    description="GeoX young academy Python example",
    author_email="",
    url="https://github.com/KDD-OpenSource/geox-young-academy/",
    keywords=["data Science", "Seminar"],
    install_requires=REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    long_description=open(os.path.join(HERE, "README.rst")).read()
)
