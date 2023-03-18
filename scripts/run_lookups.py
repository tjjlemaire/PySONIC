# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-02 17:50:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-18 18:39:47

''' Create lookup table for specific neuron. '''

import os
import itertools
import logging
import numpy as np

from PySONIC.utils import logger, isIterable, alert
from PySONIC.core import NeuronalBilayerSonophore, Batch, Lookup, AcousticDrive
from PySONIC.parsers import MechSimParser
from PySONIC.neurons import getDefaultPassiveNeuron
from PySONIC.constants import DQ_LOOKUP


# @alert
def computeAStimLookup(pneuron, aref, fref, Aref, fsref, Qref, novertones=0,
                       test=False, mpi=False, loglevel=logging.INFO):
    ''' Run simulations of the mechanical system for a multiple combinations of
        imposed sonophore radius, US frequencies, acoustic amplitudes charge densities and
        (spatially-averaged) sonophore membrane coverage fractions, compute effective
        coefficients and store them in a dictionary of n-dimensional arrays.

        :param pneuron: point-neuron model
        :param aref: array of sonophore radii (m)
        :param fref: array of acoustic drive frequencies (Hz)
        :param Aref: array of acoustic drive amplitudes (Pa)
        :param Qref: array of membrane charge densities (C/m2)
        :param fsref: acoustic drive phase (rad)
        :param mpi: boolean statting wether or not to use multiprocessing
        :param loglevel: logging level
        :return: lookups dictionary
    '''
    descs = {
        'a': 'sonophore radii',
        'f': 'US frequencies',
        'A': 'US amplitudes',
        'fs': 'sonophore membrane coverage fractions',
        'overtones': 'charge Fourier overtones'
    }

    # Populate reference vectors dictionary
    refs = {
        'a': aref,  # nm
        'f': fref,  # Hz
        'A': Aref,  # Pa
        'Q': Qref  # C/m2
    }

    err_span = 'cannot span {} for more than 1 {}'
    # If multiple sonophore coverage values, ensure that only 1 value of
    # sonophore radius and US frequency are provided
    if fsref.size > 1 or fsref[0] != 1.:
        for x in ['a', 'f']:
            assert refs[x].size == 1, err_span.format(descs['fs'], descs[x])
    # Add sonophore coverage vector to references
    refs['fs'] = fsref

    # If charge overtones are required, ensure that only 1 value of
    # sonophore radius, US frequency and coverage fraction are provided
    if novertones > 0:
        for x in ['a', 'f', 'fs']:
            assert refs[x].size == 1, err_span.format(descs['overtones'], descs[x])

    # If charge overtones are required, downsample charge and US amplitude input vectors
    if novertones > 0:
        nQmax = 50
        if len(refs['Q']) > nQmax:
            refs['Q'] = np.linspace(refs['Q'][0], refs['Q'][-1], nQmax)
        nAmax = 15
        if len(refs['A']) > nAmax:
            refs['A'] = np.insert(
                np.logspace(np.log10(refs['A'][1]), np.log10(refs['A'][-1]), num=nAmax - 1),
                0, 0.0)

    # If test case, reduce all vector dimensions to their instrinsic bounds
    if test:
        refs = {k: np.array([v.min(), v.max()]) if v.size > 1 else v for k, v in refs.items()}

    # Check validity of all reference vectors
    for key, values in refs.items():
        if not isIterable(values):
            raise TypeError(f'Invalid {descs[key]} (must be provided as list or numpy array)')
        if not all(isinstance(x, float) for x in values):
            raise TypeError(f'Invalid {descs[key]} (must all be float typed)')
        if len(values) == 0:
            raise ValueError(f'Empty {key} array')
        if key in ('a', 'f') and min(values) <= 0:
            raise ValueError(f'Invalid {descs[key]} (must all be strictly positive)')
        if key in ('A', 'fs') and min(values) < 0:
            raise ValueError(f'Invalid {descs[key]} (must all be positive or null)')

    # Create simulation queue per sonophore radius
    drives = AcousticDrive.createQueue(refs['f'], refs['A'])
    queue = []
    for drive in drives:
        for Qm in refs['Q']:
            queue.append([drive, refs['fs'], Qm])

    # Add charge overtones to queue if required
    if novertones > 0:
        # Default references
        nAQ, nphiQ = 5, 5
        AQ_ref = np.linspace(0, 100e-5, nAQ)  # C/m2
        phiQ_ref = np.linspace(0, 2 * np.pi, nphiQ, endpoint=False)  # rad
        # Downsample if test mode is on
        if test:
            AQ_ref = np.array([AQ_ref.min(), AQ_ref.max()])
            phiQ_ref = np.array([phiQ_ref.min(), phiQ_ref.max()])
        # Construct refs dict specific to Qm overtones
        Qovertones_refs = {}
        for i in range(novertones):
            Qovertones_refs[f'AQ{i + 1}'] = AQ_ref
            Qovertones_refs[f'phiQ{i + 1}'] = phiQ_ref
        # Create associated batch queue
        Qovertones = Batch.createQueue(*Qovertones_refs.values())
        Qovertones = [list(zip(x, x[1:]))[::2] for x in Qovertones]
        # Merge with main queue (moving Qm overtones into kwargs)
        queue = list(itertools.product(queue, Qovertones))
        queue = [(x[0], {'Qm_overtones': x[1]}) for x in queue]
        # Update main refs dict, and reset 'fs' as last dictionary key
        refs.update(Qovertones_refs)
        refs['fs'] = refs.pop('fs')

    # Get references dimensions
    dims = np.array([x.size for x in refs.values()])

    # Print queue (or reduced view of it)
    logger.info('batch queue:')
    Batch.printQueue(queue)

    # Run simulations and populate outputs
    logger.info('Starting simulation batch for %s neuron', pneuron.name)
    outputs = []
    for a in refs['a']:
        if pneuron.is_passive:
            xfunc = lambda *args, **kwargs: NeuronalBilayerSonophore(
                a, getDefaultPassiveNeuron()).computeEffVars(*args, **kwargs)
            batch = Batch(xfunc, queue)
        else:
            nbls = NeuronalBilayerSonophore(a, pneuron)
            batch = Batch(nbls.computeEffVars, queue)
        outputs += batch(mpi=mpi, loglevel=loglevel)

    # Split comp times and effvars from outputs
    effvars, tcomps = [list(x) for x in zip(*outputs)]
    effvars = list(itertools.chain.from_iterable(effvars))

    # Make sure outputs size matches inputs dimensions product
    nout = len(effvars)
    ncombs = dims.prod()
    if nout != ncombs:
        raise ValueError(
            f'Number of outputs ({nout}) does not match number of input combinations ({ncombs})')

    # Reshape effvars into nD arrays and add them to lookups dictionary
    logger.info(f'Reshaping {nout}-entries output into {tuple(dims)} lookup tables')
    varkeys = list(effvars[0].keys())
    tables = {}
    for key in varkeys:
        effvar = [effvars[i][key] for i in range(nout)]
        tables[key] = np.array(effvar).reshape(dims)

    # Reshape computation times, tile over extra fs dimension, and add it as a lookup table
    tcomps = np.array(tcomps).reshape(dims[:-1])
    tcomps = np.moveaxis(np.array([tcomps for i in range(dims[-1])]), 0, -1)
    tables['tcomp'] = tcomps

    # Construct and return lookup object
    return Lookup(refs, tables)


def main():

    parser = MechSimParser(outputdir='.')
    parser.addNeuron()
    parser.addTest()
    parser.defaults['neuron'] = 'RS'
    parser.defaults['radius'] = np.array([16.0, 32.0, 64.0])  # nm
    parser.defaults['freq'] = np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3])  # kHz
    parser.defaults['amp'] = np.insert(
        np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0)  # kPa
    parser.defaults['charge'] = np.nan
    parser.add_argument('--novertones', type=int, default=0, help='Number of Fourier overtones')
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:

        # Determine charge vector
        charges = args['charge']
        if charges.size == 1 and np.isnan(charges[0]):
            Qmin, Qmax = pneuron.Qbounds
            charges = np.arange(Qmin, Qmax + DQ_LOOKUP, DQ_LOOKUP)  # C/m2

        # Number of Fourier overtones
        novertones = args['novertones']

        # Determine output filename
        input_args = {'a': args['radius'], 'f': args['freq'], 'A': args['amp'], 'fs': args['fs']}
        fname_args = {k: v[0] if v.size == 1 else None for k, v in input_args.items()}
        fname_args['novertones'] = novertones
        lookup_fpath = NeuronalBilayerSonophore(32e-9, pneuron).getLookupFilePath(**fname_args)

        # Combine inputs into single list
        inputs = [args[x] for x in ['radius', 'freq', 'amp', 'fs']] + [charges]

        # Adapt inputs and output filename if test case
        if args['test']:
            fcode, fext = os.path.splitext(lookup_fpath)
            lookup_fpath = f'{fcode}_test{fext}'

        # Check if lookup file already exists
        if os.path.isfile(lookup_fpath):
            logger.warning(
                f'"{lookup_fpath}" file already exists and will be overwritten. Continue? (y/n)')
            user_str = input()
            if user_str not in ['y', 'Y']:
                logger.error('%s Lookup creation canceled', pneuron.name)
                return

        # Compute lookup
        lkp = computeAStimLookup(pneuron, *inputs, novertones=novertones,
                                 test=args['test'], mpi=args['mpi'], loglevel=args['loglevel'])
        logger.info(f'Generated lookup: {lkp}')

        # Save lookup in PKL file
        logger.info('Saving %s neuron lookup in file: "%s"', pneuron.name, lookup_fpath)
        lkp.toPickle(lookup_fpath)


if __name__ == '__main__':
    main()
