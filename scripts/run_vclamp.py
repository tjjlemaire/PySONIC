# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-14 11:56:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-17 17:06:04

from PySONIC.core import Batch, VoltageClamp
from PySONIC.utils import logger
from PySONIC.parsers import VClampParser

def main():
    # Parse command line arguments
    parser = VClampParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run E-STIM batch
    logger.info("Starting V-clamp simulation batch")
    queue = VoltageClamp.simQueue(
        *parser.parseSimInputs(args), outputdir=args['outputdir'], overwrite=args['overwrite'])
    output = []
    for pneuron in args['neuron']:
        vlcamp = VoltageClamp(pneuron)
        batch = Batch(vlcamp.simAndSave if args['save'] else vlcamp.simulate, queue)
        output += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)

if __name__ == '__main__':
    main()