#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [ ]


setup(
    author="Themis",
    author_email='poultourd@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Good_question',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11'
    ],
    description="A package of small widely useful (or not) functions that i haev written and want to be able to use easily from my computer",
    #install_requires=requirements,
    long_description=readme + '\n\n' + 'history, but i did not include',
    #include_package_data=True,
    keywords='odos',
    name='odos',
    packages=find_packages(include=['odos', 'odos.*']),
)
