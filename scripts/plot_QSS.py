# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-03 21:38:46

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import plotQSSDerivativeVsState, plotQSSVarVsQm, plotEqChargeVsAmp, plotQSSThresholdCurve
from PySONIC.parsers import AStimParser


def main():

    # Parse command line arguments
    parser = AStimParser()
    # parser.addCmap(default='viridis')
    parser.addAscale()
    parser.outputdir_dep_key = 'save'
    parser.addInputDir(dep_key='compare')
    parser.defaults['amp'] = np.logspace(np.log10(1), np.log10(600), 100)  # kPa
    parser.defaults['tstim'] = 1000.  # ms
    parser.defaults['toffset'] = 0.  # ms
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['plot'] is None:
        args['plot'] = ['dQdt']
    a, f, tstim, toffset, PRF = [
        args[k][0] for k in ['radius', 'freq', 'tstim', 'toffset', 'PRF']]
    qss_vars = args['qss']

    figs = []
    # For each neuron type
    for i, pneuron in enumerate(args['neuron']):

        # If only 1 DC value
        if args['DC'].size == 1:
            DC = args['DC'][0]

            # If only 1 amplitude value
            if args['amp'].size == 1:

                # Plot QSS derivative vs state for specified variables
                if qss_vars is not None:
                    for k in qss_vars:
                        figs.append(plotQSSDerivativeVsState(pneuron, a, f, args['amp'][0], DC))
            else:
                # Plot evolution of QSS vars vs Q for different amplitudes
                # for pvar in args['plot']:
                #     figs.append(plotQSSVarVsQm(
                #         pneuron, a, f, pvar, amps=args['amp'], DC=DC,
                #         cmap=args['cmap'], zscale=args['Ascale'], mpi=args['mpi'],
                #         loglevel=args['loglevel']))

                # Plot equilibrium charge as a function of amplitude
                if 'dQdt' in args['plot']:
                    figs.append(plotEqChargeVsAmp(
                        pneuron, a, f, amps=args['amp'], tstim=tstim, toffset=toffset, PRF=PRF,
                        DC=DC, xscale=args['Ascale'], compdir=args['inputdir'], mpi=args['mpi'],
                        loglevel=args['loglevel']))
        else:
            figs.append(plotQSSThresholdCurve(
                pneuron, a, f, tstim=tstim, toffset=toffset, PRF=PRF, DCs=args['DC'],
                Ascale=args['Ascale'], comp=args['compare'], mpi=args['mpi'],
                loglevel=args['loglevel']))

    if args['save']:
        for fig in figs:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            fig.savefig(os.path.join(args['outputdir'], f'{s}.png'), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
