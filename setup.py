#!/usr/bin/env python3
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='K_submodular',
    version='1.0.0',
    description='Description here',
    long_description_content_type='text/markdown',
    author='Author',

    project_urls={
        'Bug Reports': '',
        'Source': '',
    },
    keywords='k_submodular, submodular',

    classifiers=[

        ],

    package_dir={'': './'},
    packages=find_packages(where='./'),

    include_package_data=True,

    python_requires='>=3.7, <4',
    install_requires=[

        ],

    entry_points={

    }
)