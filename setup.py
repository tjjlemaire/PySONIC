# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-13 09:40:02
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-28 17:01:43

import os
from setuptools import setup

readme_file = 'README.md'


def readme():
    with open(readme_file, encoding="utf8") as f:
        return f.read()


def description():
    with open(readme_file, encoding="utf8") as f:
        started = False
        lines = []
        for line in f:
            if not started:
                if line.startswith('# Description'):
                    started = True
            else:
                if line.startswith('#'):
                    break
                else:
                    lines.append(line)
    return ''.join(lines).strip('\n')


def getFiles(path):
    return [f'{path}/{x}' for x in os.listdir(path)]


setup(
    name='PySONIC',
    version='1.0',
    description=description(),
    long_description=readme(),
    url='https://iopscience.iop.org/article/10.1088/1741-2552/ab1685',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords=('SONIC NICE acoustic ultrasound ultrasonic neuromodulation neurostimulation excitation\
             computational model intramembrane cavitation'),
    author='Théo Lemaire',
    author_email='theo.lemaire@epfl.ch',
    license='MIT',
    packages=['PySONIC'],
    scripts=getFiles('scripts') + getFiles('tests'),
    install_requires=[
        'numpy>=1.10',
        'scipy>=0.17',
        'matplotlib>=2',
        'pandas>=1.2.4',
        'colorlog>=3.0.1',
        'tqdm>=4.3',
        'lockfile>=0.1.2',
        'multiprocess>=0.70',
        'pushbullet.py>=0.11.0',
        'boltons>=20.1.0'
    ],
    zip_safe=False
)
