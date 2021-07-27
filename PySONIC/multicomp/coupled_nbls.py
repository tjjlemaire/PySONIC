# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 17:50:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-20 16:30:47

import os
import pickle
import logging
import numpy as np

from ..utils import logger, isWithin, os_name
from ..core import Model, NeuronalBilayerSonophore, EventDrivenSolver
from ..core.timeseries import TimeSeries, SpatiallyExtendedTimeSeries
from ..constants import CLASSIC_TARGET_DT, MAX_NSAMPLES_EFFECTIVE


class CoupledSonophores:
    ''' Interface allowing to run benchmark simulations of a two-compartment coupled NBLS model. '''

    simkey = 'COUPLED_ASTIM'  # keyword used to characterize simulations made with this model
    ga_bounds = [1e-10, 1e10]  # S/m2

    def __init__(self, nodes, ga):
        ''' Initialization.

            :param nodes: list of nbls objects
            :param ga: axial conductance (S/m2)
        '''
        assert all(x.pneuron == nodes[0].pneuron for x in nodes), 'differing point-neuron models'
        self.nodes = nodes
        self.nnodes = len(nodes)
        self.ga = ga

    def normalizedConductanceMatrix(self):
        ones = np.ones(self.nnodes)
        return np.diag(ones, 0) + np.diag(-ones[:-1], -1) + np.diag(-ones[:-1], 1)

    def copy(self):
        return self.__class__(self.nodes, self.ga)

    @property
    def meta(self):
        return {
            'nodes': [x.meta for x in self.nodes],
            'ga': self.ga
        }

    @classmethod
    def initFromMeta(cls, meta):
        try:
            nodes, ga = meta['nodes'], meta['ga']
        except KeyError:
            meta = meta['model']
            nodes, ga = meta['nodes'], meta['ga']
        nodes = [NeuronalBilayerSonophore.initFromMeta(x) for x in nodes]
        return cls(nodes, ga)

    @property
    def refnode(self):
        return self.nodes[0]

    @property
    def refpneuron(self):
        return self.refnode.pneuron

    @property
    def gastr(self):
        return f'{self.ga:.2e} S/m2'

    @property
    def mechstr(self):
        return self.refnode.pneuron.name

    def __repr__(self):
        params = [f'ga = {self.gastr}']
        return f'{self.__class__.__name__}({self.mechstr} dynamics, {", ".join(params)})'

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, value):
        if value != 0.:
            assert isWithin('ga', value, self.ga_bounds)
        self._ga = value
        self.ga_matrix = self.normalizedConductanceMatrix() * value

    def Iax(self, Vm):
        ''' Compute array of axial currents in each compartment based on array of potentials. '''
        return -self.ga_matrix.dot(Vm)  # mA/m2

    def serialize(self, y):
        ''' Serialize a single state vector into a state-per-node matrix. '''
        return np.reshape(y.copy(), (self.npernode * self.nnodes))

    def deserialize(self, y):
        ''' Deserialize a state per node matrix into a single state vector. '''
        return np.reshape(y.copy(), (self.nnodes, self.npernode))

    def fullDerivatives(self, t, y, drives, fs):
        ''' full derivatives method. '''
        # Deserialize states vector
        y = self.deserialize(y)
        # Compute derivatives array for uncoupled nbls systems
        dydt = np.vstack([self.nodes[i].fullDerivatives(t, y[i], drives[i], fs[i])
                          for i in range(self.nnodes)])
        # Compute membrane capacitance and potential profiles
        iZ, iQ = 1, 3
        Cm = np.array([x.spatialAverage(fs[i], x.capacitance(y[i, iZ]), x.Cm0)
                       for i, x in enumerate(self.nodes)])  # F/m2
        Vm = y[:, iQ] / Cm * 1e3  # mV
        # Add axial currents to charge derivatives column
        dydt[:, iQ] += self.Iax(Vm) * 1e-3  # A/m2
        # Return serialized derivatives vector
        return self.serialize(dydt)

    def effDerivatives(self, t, y, lkps1d):
        ''' effective derivatives method. '''
        # Deserialize states vector
        y = self.deserialize(y)
        # Compute derivatives array for uncoupled nbls systems
        iQ = 0
        dydt = np.vstack([self.nodes[i].effDerivatives(t, y[i], lkps1d[i], [])
                          for i in range(self.nnodes)])
        # Interpolate membrane potentials
        Vm = np.array([x.interpolate1D(Qm)['V'] for x, Qm in zip(lkps1d, y[:, iQ])])  # mV
        # Add axial currents to charge derivatives column
        dydt[:, iQ] += self.Iax(Vm) * 1e-3  # A/m2
        # Return serialized derivatives vector
        return self.serialize(dydt)

    def deserializeSolution(self, data):
        ''' Re-arrange solution per node. '''
        inputs = [data.time, data.stim]
        output_keys = {
            f'node{i + 1}': list(filter(lambda x: x.endswith(f'_{i + 1}'), data.outputs))
            for i in range(self.nnodes)}
        outputs = {k: {x[:-2]: data[x].values for x in v} for k, v in output_keys.items()}
        return SpatiallyExtendedTimeSeries({
            k: TimeSeries(*inputs, v) for k, v in outputs.items()})

    def __simFull(self, drives, pp, fs):
        # Determine time step
        assert drives.is_monofrequency(), 'differing carrier frequencies'
        dt = drives.dt

        # Compute and serialize initial conditions
        y0 = [self.nodes[i].fullInitialConditions(drives[i], self.nodes[i].Qm0, dt)
              for i in range(self.nnodes)]
        self.npernode = len(y0[0])
        y0 = {f'{k}_{i + 1}': v for i, x in enumerate(y0) for k, v in x.items()}

        # Define event function
        def updateDrives(obj, x):
            for sd, d in zip(obj.drives, drives):
                sd.xvar = d.xvar * x

        # Initialize solver
        solver = EventDrivenSolver(
            lambda x: updateDrives(solver, x),
            y0.keys(),
            lambda t, y: self.fullDerivatives(t, y, solver.drives, fs),
            event_params={'drives': drives.nullCopy()},
            dt=dt)

        # Compute serialized solution
        data = solver(
            y0, pp.stimEvents(), pp.tstop, target_dt=CLASSIC_TARGET_DT,
            log_period=pp.tstop / 100 if logger.getEffectiveLevel() <= logging.INFO else None,
            # logfunc=lambda y: f'Qm = {y[3] * 1e5:.2f} nC/cm2'
        )

        # Re-arrange solution per node
        data = self.deserializeSolution(data)

        # Remove velocity and add voltage timeseries to solution
        for i, (nodekey, nodedata) in enumerate(data.items()):
            del nodedata['U']
            nodedata.addColumn(
                'Vm', self.nodes[i].deflectionDependentVm(nodedata['Qm'], nodedata['Z'], fs[i]))

        # Return solution
        return data

    def __simSonic(self, drives, pp, fs):
        # Determine time step
        assert drives.is_monofrequency(), 'differing carrier frequencies'
        dt = drives.periodicity  # s

        # Load appropriate 2D lookups
        lkps = [self.nodes[i].getLookup2D(drives[i].f, fs[i]) for i in range(self.nnodes)]

        # Compute and serialize initial conditions
        y0 = [{'Qm': x.Qm0, **{k: v(x.pneuron.Vm0) for k, v in x.pneuron.steadyStates().items()}}
              for x in self.nodes]
        self.npernode = len(y0[0])
        y0 = {f'{k}_{i + 1}': v for i, x in enumerate(y0) for k, v in x.items()}

        # Define event function
        def updateLookups(obj, x):
            for i, (lkp, drive) in enumerate(zip(lkps, drives)):
                obj.lkps[i] = lkp.project('A', drive.xvar * x)

        # Initialize solver
        solver = EventDrivenSolver(
            lambda x: updateLookups(solver, x),
            y0.keys(),
            lambda t, y: self.effDerivatives(t, y, solver.lkps),
            event_params={'lkps': [lkp.project('A', 0.) for lkp in lkps]},
            dt=dt)

        # Compute serialized solution
        data = solver(
            y0, pp.stimEvents(), pp.tstop,
            log_period=pp.tstop / 100 if pp.tstop >= 5 else None,
            max_nsamples=MAX_NSAMPLES_EFFECTIVE)

        # Re-arrange solution per node
        data = self.deserializeSolution(data)

        # Add voltage timeseries to solution (interpolated along Qm) and dummy Z and ng vectors
        for i, (nodekey, nodedata) in enumerate(data.items()):
            nodedata.addColumn('Vm', self.nodes[i].interpEffVariable(
                'V', nodedata['Qm'], nodedata.stim * drives[i].A, lkps[i]))
            for key in ['Z', 'ng']:
                nodedata[key] = np.full(nodedata.time.size, np.nan)

        # Return solution dataframe
        return data

    def intMethods(self):
        ''' Listing of model integration methods. '''
        return {
            'full': self.__simFull,
            'sonic': self.__simSonic
        }

    def desc(self, meta):
        method = meta['method'] if 'method' in meta else meta['model']['method']
        fs = meta['fs'] if 'fs' in meta else meta['model']['fs']
        drives_str = meta['drives'].desc
        fs_str = f'fs = ({", ".join([f"{x * 1e2:.2f}%" for x in fs])})'
        return f'{self}: {method} simulation @ ({drives_str}), {meta["pp"].desc}, {fs_str}'

    @Model.addMeta
    @Model.logDesc
    def simulate(self, drives, pp, fs, method='sonic'):
        ''' Simulate the coupled-nbls model for a specific set of US stimulation parameters,
            and coverage fractions, and return spatially-distributed output data.

            :param drives: acoustic drive array
            :param pp: pulse protocol object
            :param fs: list of sonophore membrane coverage fractions (-)
            :param method: selected integration method
            :return: SpatiallyExtendedTimeSeries object
        '''
        # Check that inputs dimensions matches number of nodes
        assert len(drives) == self.nnodes, 'number of drives does not match number of nodes'
        assert len(fs) == self.nnodes, 'number of coverage inputs does not match number of nodes'
        simfunc = self.intMethods()[method]
        return simfunc(drives, pp, fs)

    def filecodes(self, drives, pp, fs, method):
        codes = {
            'simkey': self.simkey,
            'neuron': self.refpneuron.name,
            'nnodes': f'{self.nnodes}node{"s" if self.nnodes > 1 else ""}',
            'ga': f'ga{self.ga:.2e}S_m2',
            'a': f'a{"_".join([f"{x.a * 1e9:.0f}nm" for x in self.nodes])}',
            **drives.filecodes,
            **pp.filecodes
        }
        codes['fs'] = f'fs{"_".join([f"{x * 1e2:.0f}%" for x in fs])}'
        codes['method'] = method
        return codes

    def filecode(self, *args):
        return '_'.join([x for x in self.filecodes(*args).values() if x is not None])

    def simAndSave(self, *args, outdir=None, overwrite=False, full_output=False):
        runsim = True
        if outdir is not None:
            fname = f'{self.filecode(*args)}.pkl'
            fpath = os.path.join(outdir, fname)
            if os_name == 'Windows':
                fpath = f'\\\\?\\{fpath}'
            if os.path.isfile(fpath) and not overwrite:
                logger.info(f'Loading data from "{os.path.basename(fpath)}"')
                with open(fpath, 'rb') as fh:
                    frame = pickle.load(fh)
                    data, meta = frame['data'], frame['meta']
                    runsim = False
        if runsim:
            data, meta = self.simulate(*args)
            if not full_output:
                data.dumpOutputsOtherThan(['Qm', 'Vm'])
            if outdir is not None:
                with open(fpath, 'wb') as fh:
                    pickle.dump({'meta': meta, 'data': data}, fh)
                logger.debug(f'simulation data exported to "{fpath}"')
        return data, meta

    @property
    def tauax(self):
        ''' Axial time constant (s). '''
        return self.refnode.Cm0 / self.ga

    @property
    def taum(self):
        ''' Passive membrane time constant (s). '''
        return self.refpneuron.tau_pas

    @property
    def taumax(self):
        ''' Maximal time constant of the model (s). '''
        return max(self.taum, self.tauax)
