# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-11-21 10:46:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-17 17:05:24

''' Run simulations of the NICE mechanical model. '''

from PySONIC.core import BilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.parsers import MechSimParser


def main():
    # Parse command line arguments
    parser = MechSimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run MECH batch
    logger.info("Starting mechanical simulation batch")
    queue = BilayerSonophore.simQueue(
        *parser.parseSimInputs(args), outputdir=args['outputdir'], overwrite=args['overwrite'])
    output = []
    for a in args['radius']:
        for d in args['embedding']:
            for Cm0 in args['Cm0']:
                for Qm0 in args['Qm0']:
                    bls = BilayerSonophore(a, Cm0, Qm0, embedding_depth=d)
                    batch = Batch(bls.simAndSave if args['save'] else bls.simulate, queue)
                    output += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)


if __name__ == '__main__':
    main()
