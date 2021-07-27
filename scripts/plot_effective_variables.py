# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-15 15:59:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-03 21:21:58

''' Plot the effective variables as a function of charge density with color code. '''

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger
from PySONIC.parsers import MechSimParser

# Set logging level
logger.setLevel(logging.INFO)


def main():

    parser = MechSimParser()
    parser.addNeuron()
    parser.addNColumns()
    parser.addNLevels()
    parser.defaults['neuron'] = 'RS'
    parser.defaults['radius'] = np.nan
    parser.defaults['freq'] = np.nan
    parser.defaults['amp'] = np.nan
    args = parser.parse()
    for k in ['charge', 'embedding', 'Cm0', 'Qm0', 'fs', 'mpi', 'pltscheme', 'plot']:
        del args[k]
    logger.setLevel(args['loglevel'])

    # Restrict radius, frequency and amplitude to single values
    for k in ['radius', 'freq', 'amp']:
        if len(args[k]) > 1:
            logger.error(f'multiple {k} values not allowed')
        val = args[k][0]
        if np.isnan(val):
            args[k] = None
        else:
            args[k] = val

    for pneuron in args['neuron']:
        plotEffectiveVariables(
            pneuron, a=args['radius'], f=args['freq'], A=args['amp'],
            zscale=args['cscale'], cmap=args['cmap'], ncolmax=args['ncol'], nlevels=args['nlevels'])

    plt.show()


if __name__ == '__main__':
    main()
