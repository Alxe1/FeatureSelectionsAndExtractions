# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : setup.py
# Description:
from setuptools import setup, find_packages

setup(
    name="fsae",
    version="1.0.0",
    description="feature selections and feature extractions",
    author="Littlely",
    packages=find_packages(),
    install_requires=[
        "numpy==1.22.0",
        "scikit-learn==0.20.2"
    ]
)