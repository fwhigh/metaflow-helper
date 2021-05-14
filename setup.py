#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
import pkg_resources

# Package meta-data.
NAME = 'metalearn'
DESCRIPTION = 'Convenience utilities for common machine learning tasks on Metaflow'
URL = 'https://github.com/fwhigh/metalearn'
EMAIL = 'fwhigh@gmail.com'
AUTHOR = 'F. William High'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'
LICENSE = 'Apache-2.0'

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'requirements.txt'), 'r') as requirements_txt:
    REQUIRED = [
        str(req)
        for req in pkg_resources.parse_requirements(requirements_txt)
    ]

EXTRAS = {
    # 'fancy feature': ['django'],
}


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        f"License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords='circleci ci cd api sdk',
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
