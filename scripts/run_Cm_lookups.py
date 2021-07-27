# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-05 17:42:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-05 20:31:11

''' Create Cm lookup table. '''

import os
import logging
import numpy as np

from PySONIC.utils import logger, isIterable, alert
from PySONIC.core import BilayerSonophore, Batch, Lookup, AcousticDrive
from PySONIC.parsers import MechSimParser


@alert
def computeCmLookup(bls, fref, Aref, mpi=False, loglevel=logging.INFO):
    descs = {'f': 'US frequencies', 'A': 'US amplitudes'}

    # Populate reference vectors dictionary
    refs = {
        'f': fref,  # Hz
        'A': Aref   # Pa
    }

    # Check validity of all reference vectors
    for key, values in refs.items():
        if not isIterable(values):
            raise TypeError(f'Invalid {descs[key]} (must be provided as list or numpy array)')
        if not all(isinstance(x, float) for x in values):
            raise TypeError(f'Invalid {descs[key]} (must all be float typed)')
        if len(values) == 0:
            raise ValueError(f'Empty {key} array')
        if key == 'f' and min(values) <= 0:
            raise ValueError(f'Invalid {descs[key]} (must all be strictly positive)')
        if key == 'A' and min(values) < 0:
            raise ValueError(f'Invalid {descs[key]} (must all be positive or null)')

    # Get references dimensions
    dims = np.array([x.size for x in refs.values()])

    # Create simulation queue
    drives = AcousticDrive.createQueue(fref, Aref)
    queue = [[drive, 0.] for drive in drives]

    # Run simulations and populate outputs
    logger.info(f'Starting Cm simulation batch for {bls}')
    batch = Batch(bls.getRelCmCycle, queue)
    rel_Cm_cycles = batch(mpi=mpi, loglevel=loglevel)

    # Make sure outputs size matches inputs dimensions product
    nout, nsamples = len(rel_Cm_cycles), rel_Cm_cycles[0].size
    assert nout == dims.prod(), 'Number of outputs does not match number of combinations'
    dims = np.hstack([dims, nsamples])
    refs['t'] = np.linspace(0., 1., nsamples)

    # Reshape effvars into nD arrays and add them to lookups dictionary
    logger.info('Reshaping output into lookup table')
    rel_Cm_cycles = np.array(rel_Cm_cycles).reshape(dims)

    # Construct and return lookup object
    return Lookup(refs, {'Cm_rel': rel_Cm_cycles})


def main():

    parser = MechSimParser(outputdir='.')
    parser.addTest()
    parser.defaults['radius'] = 32.0  # nm
    parser.defaults['freq'] = np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3])  # kHz
    parser.defaults['amp'] = np.insert(
        np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0)  # kPa
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Model
    a = args['radius'][0]  # m
    Cm0 = 1e-2             # F/m2
    Qm0 = 0.0              # C/m2
    bls = BilayerSonophore(a, Cm0, Qm0)
    lookup_fpath = bls.Cm_lkp_filepath

    # Batch inputs
    inputs = [args[x] for x in ['freq', 'amp']]

    # Adapt inputs and output filename if test case
    if args['test']:
        for i, x in enumerate(inputs):
            if x is not None and x.size > 1:
                inputs[i] = np.array([x.min(), x.max()])
        fcode, fext = os.path.splitext(lookup_fpath)
        lookup_fpath = f'{fcode}_test{fext}'

    # Check if lookup file already exists
    if os.path.isfile(lookup_fpath):
        logger.warning(
            f'"{lookup_fpath}" file already exists and will be overwritten. Continue? (y/n)')
        user_str = input()
        if user_str not in ['y', 'Y']:
            logger.error('Cm-lookup creation canceled')
            return

    # Compute lookup
    lkp = computeCmLookup(bls, *inputs, mpi=args['mpi'], loglevel=args['loglevel'])
    logger.info(f'Generated Cm-lookup: {lkp}')

    # Save lookup in PKL file
    logger.info(f'Saving {bls} Cm-lookup in file: "{lookup_fpath}"')
    lkp.toPickle(lookup_fpath)


if __name__ == '__main__':
    main()
