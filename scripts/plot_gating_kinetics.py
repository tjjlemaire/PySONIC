# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-10-11 20:35:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-26 16:11:03

''' Plot the voltage-dependent steady-states and time constants of activation and inactivation
    gates of the different ionic currents involved in the neuron's membrane dynamics. '''


import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import plotGatingKinetics
from PySONIC.parsers import Parser


def main():
    parser = Parser()
    parser.addNeuron()
    parser.addYscale()  # only for tau axis
    parser.defaults['neuron'] = 'RS'
    parser.defaults['yscale'] = 'lin'
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:
        plotGatingKinetics(pneuron, tau_scale=args['yscale'])
    plt.show()


if __name__ == '__main__':
    main()
