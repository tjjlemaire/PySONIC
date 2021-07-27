# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-13 18:10:20

''' Plot (duty-cycle x amplitude) US activation map of a neuron at a given frequency and PRF. '''

import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import getActivationMap
from PySONIC.parsers import AStimParser


def main():

    # Parse command line arguments
    parser = AStimParser()
    parser.defaults['amp'] = np.logspace(np.log10(10), np.log10(600), 30)  # kPa
    parser.defaults['DC'] = np.arange(1, 101)  # %
    parser.defaults['tstim'] = 200.  # ms
    parser.defaults['toffset'] = 0.  # ms
    parser.addInputDir()
    parser.addThresholdCurve()
    parser.addInteractive()
    parser.addYscale()
    parser.addZvar('FR')
    parser.addZscale()
    parser.addZbounds((1e0, 1e3))
    parser.outputdir_dep_key = 'save'
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:
        for a in args['radius']:
            for fs in args['fs']:
                for f in args['freq']:
                    for tstim in args['tstim']:
                        for PRF in args['PRF']:
                            actmap = getActivationMap(
                                args['zvar'], args['inputdir'], pneuron, a, fs,
                                f, tstim, PRF, args['amp'], args['DC'])
                            actmap.run(mpi=args['mpi'])
                            actmap.render(
                                cmap=args['cmap'],
                                yscale=args['yscale'],
                                zscale=args['zscale'],
                                zbounds=args['zbounds'],
                                interactive=args['interactive'],
                                thresholds=args['threshold'],
                            )

    plt.show()


if __name__ == '__main__':
    main()
