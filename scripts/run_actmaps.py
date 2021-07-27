# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-09 10:52:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-18 18:10:13

''' Example script showing how to build neuron activation maps. '''

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getPointNeuron
from PySONIC.plt import getActivationMap

logger.setLevel(logging.INFO)


def main():
    ''' The code must be wrappped inside a main function in order to allow MPI usage. '''
    # Parameters
    root = selectDirDialog()
    pneuron = getPointNeuron('TC')
    a = 32e-9  # m
    coverages = [0.8, 1.0]
    f = 500e3  # Hz
    tstim = 100e-3  # s
    PRF = 100  # Hz
    amps = np.logspace(np.log10(10.), np.log10(600.), 5) * 1e3  # Pa
    DCs = np.linspace(5, 100, 5) * 1e-2  # (-)

    # Define variable-dependent z-bounds and colormaps
    zbounds = {
        'FR': (1e0, 1e3),  # Hz
        'Cai': (1e0, 1e2)  # uM
    }
    cmap = {'FR': 'viridis', 'Cai': 'cividis'}

    # For each coevrage fraction
    for fs in coverages:
        # For each map class
        for zkey in ['FR', 'Cai']:
            # Create activation map object
            actmap = getActivationMap(zkey, root, pneuron, a, fs, f, tstim, PRF, amps, DCs)

            # Run simulations for populate the 2D map
            actmap.run(mpi=True)

            # Render the 2D map
            actmap.render(interactive=True, zbounds=zbounds[zkey], cmap=cmap[zkey])

    plt.show()


if __name__ == '__main__':
    main()
