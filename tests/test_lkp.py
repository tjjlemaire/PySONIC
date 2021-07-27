# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-07 17:14:17
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-26 12:36:20

import numpy as np
from PySONIC.core import EffectiveVariablesLookup

''' Test the lookup functionalities. '''

# Define 4 reference dimensions with different respective sizes
refs = {
    'a': np.logspace(np.log10(16), np.log10(64), 5),
    'f': np.linspace(100, 4e3, 6),
    'A': np.hstack(([0], np.logspace(np.log10(1), np.log10(600), 7))),
    'Q': np.linspace(-80, 50, 3)
}

# Define corresponding 4D tables
dims = [refs[x].shape[0] for x in refs.keys()]
tables = { k: np.random.rand(*dims) for k in ['alpham', 'betam']}

########### Generic lookup features  ###########

# Initialization
lkp4d = EffectiveVariablesLookup(refs, tables)
print('initialization:', lkp4d)
print()

# Simultaneous projection at specific values in the first 3 dimensions
lkp1d = lkp4d.projectN({'a': 32., 'f': 500., 'A': 100.})
print('after projection:', lkp1d)
for k, v in lkp1d.items():
    print(f'   {k}: {v}')
print()

# Algebraic operations on lookup
lkp1d = lkp1d * lkp1d + 3
print('after algebraic operations:', lkp1d)
for k, v in lkp1d.items():
    print(f'   {k}: {v}')
print()

# Tiling in extra dimension of size 2
lkp2d = lkp1d.tile('extra', np.array([1, 2]))
print('after tiling:', lkp2d)
for k, v in lkp2d.items():
    print(f'   {k}: {v}')
print()


########### Smart lookup features  ###########

lkp4d = EffectiveVariablesLookup(refs, tables)
print('initialization:', lkp4d)
print()

# DC projection
lkp4d = lkp4d.projectDC(DC=0.4)
print('after DC projection:', lkp4d)
print()

# Derived tables
print('derived tables:')
for k in ['taum', 'minf']:
    print(f'   {k}:', lkp1d[k])
print()

