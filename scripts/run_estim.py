# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-17 20:12:56

''' Run E-STIM simulations of a specific point-neuron. '''

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.parsers import EStimParser


def main():
    # Parse command line arguments
    parser = EStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    sim_inputs = parser.parseSimInputs(args)
    simQueue_func = {5: 'simQueue', 6: 'simQueueBurst'}[len(sim_inputs)]

    # Run E-STIM batch
    logger.info("Starting E-STIM simulation batch")
    queue = getattr(PointNeuron, simQueue_func)(
        *sim_inputs, outputdir=args['outputdir'], overwrite=args['overwrite'])
    output = []
    for pneuron in args['neuron']:
        batch = Batch(pneuron.simAndSave if args['save'] else pneuron.simulate, queue)
        output += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()
