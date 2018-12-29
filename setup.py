#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'torch>=1.0.0',
    'torchvision>=0.2.1',
    'fs',
    'scipy', 'numpy', 'attrs', 'tqdm', 'matplotlib',
    'tensorboardX'
]

setup_requirements = []

test_requirements = [
    'nose',
    'mypy',
    'pylint',
    'flake8',
]

setup(
    author="Richard Weiss",
    author_email='richardweiss@richardweiss.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Cycle gan implementation",
    entry_points={
        'console_scripts': [
            'cyclegan=cyclegan.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cyclegan',
    name='cyclegan',
    packages=find_packages('src', include=['cyclegan']),
    package_dir={"": "src"},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ririw/cyclegan',
    version='0.1.0',
    zip_safe=False,
)
