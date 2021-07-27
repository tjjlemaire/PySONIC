# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-20 16:27:31

import os
import numpy as np
import matplotlib.pyplot as plt

from ..core import NeuronalBilayerSonophore, PulsedProtocol, Batch
from ..core.drives import AcousticDrive, AcousticDriveArray
from ..utils import si_format, rmse, rescale, logger, bounds
from ..neurons import passiveNeuron
from ..postpro import gamma
from ..plt import harmonizeAxesLimits, hideSpines, hideTicks, addYscale, addXscale
from .coupled_nbls import CoupledSonophores


class Benchmark:

    tsparse_bounds = (1, -2)

    def __init__(self, a, nnodes, outdir=None, nodecolors=None):
        self.a = a
        self.nnodes = nnodes
        self.outdir = outdir
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if nodecolors is None:
            nodecolors = plt.get_cmap('Dark2').colors
        self.nodecolors = nodecolors

    def pdict(self):
        return {
            'a': f'{self.a * 1e9:.0f} nm',
            'nnodes': f'{self.nnodes} nodes',
        }

    def pstr(self):
        l = []
        for k, v in self.pdict().items():
            if k == 'nnodes':
                l.append(v)
            else:
                l.append(f'{k} = {v}')
        return ', '.join(l)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pstr()})'

    def code(self):
        s = self.__repr__()
        for k in ['/', '(', ',']:
            s = s.replace(k, '_')
        for k in ['=', ' ', ')']:
            s = s.replace(k, '')
        return s

    def runSims(self, model, drives, tstim, covs):
        ''' Run full and sonic simulations for a specific combination drives,
            pulsed protocol and coverage fractions, and harmonize outputs.
        '''
        Fdrive = drives[0].f
        assert all(x.f == Fdrive for x in drives), 'frequencies do not match'
        assert len(covs) == model.nnodes, 'coverages do not match model dimensions'
        assert len(drives) == model.nnodes, 'drives do not match model dimensions'

        # If not provided, compute stimulus duration from model passive properties
        min_ncycles = 10
        ntaumax_conv = 5
        if tstim is None:
            tstim = max(ntaumax_conv * model.taumax, min_ncycles / Fdrive)
        # Recast stimulus duration as finite multiple of acoustic period
        tstim = int(np.ceil(tstim * Fdrive)) / Fdrive  # s

        # Pulsed protocol
        pp = PulsedProtocol(tstim, 0)

        # Simulate/Load with full and sonic methods
        data, meta = {}, {}
        for method in ['full', 'sonic']:
            data[method], meta[method] = model.simAndSave(
                drives, pp, covs, method, outdir=self.outdir,
                overwrite=False, full_output=False)

        # Cycle-average full solution and interpolate sonic solution along same time vector
        data['cycleavg'] = data['full'].cycleAveraged(1 / Fdrive)
        data['sonic'] = data['sonic'].interpolate(data['cycleavg'].time)

        # # Compute normalized charge density profiles and add them to dataset
        # for simkey, simdata in data.items():
        #     for nodekey, nodedata in simdata.items():
        #         nodedata['Qnorm'] = nodedata['Qm'] / model.refpneuron.Cm0 * 1e3  # mV

        # Return dataset
        return data, meta

    def getTime(self, data):
        ''' Get time vector compatible with cycle-averaged and sonic charge density vectors
            (with, by default, discarding of bounding artefact elements).
        '''
        return data['cycleavg'].time[self.tsparse_bounds[0]:self.tsparse_bounds[1]]

    def getCharges(self, data, k, cut_bounds=True):
        ''' Get node-specific list of cycle-averaged and sonic charge density vectors
            (with, by default, discarding of bounding artefact elements).
        '''
        Qms = np.array([data[simkey][k]['Qm'].values for simkey in ['cycleavg', 'sonic']])
        if cut_bounds:
            Qms = Qms[:, self.tsparse_bounds[0]:self.tsparse_bounds[1]]
        return Qms

    def computeRMSE(self, data):
        ''' Evaluate per-node RMSE on charge density profiles. '''
        return {k: rmse(*self.getCharges(data, k)) for k in data['cycleavg'].keys()}

    def eval_funcs(self):
        return {
            'rmse': (self.computeRMSE, 'nC/cm2')
        }

    def computeDivergence(self, data, eval_mode, *args):
        ''' Compute divergence according to given eval_mode, and return max value across nodes. '''
        divs = list(self.eval_funcs()[eval_mode][0](data, *args).values())
        if any(np.isnan(x) for x in divs):
            return np.nan
        return max(divs)

    def plotQm(self, ax, data):
        ''' Plot charge density signals on an axis. '''
        markers = {'full': '-', 'cycleavg': '--', 'sonic': '-'}
        alphas = {'full': 0.5, 'cycleavg': 1., 'sonic': 1.}
        # tplt = TimeSeriesPlot.getTimePltVar('ms')
        # yplt = self.model.refpneuron.getPltVars()['Qm/Cm0']
        # mode = 'details'
        for simkey, simdata in data.items():
            for i, (nodekey, nodedata) in enumerate(simdata.items()):
                y = nodedata['Qm'].values
                y[-1] = y[-2]
                ax.plot(nodedata.time * 1e3, y * 1e5, markers[simkey], c=self.nodecolors[i],
                        alpha=alphas[simkey], label=f'{simkey} - {nodekey}')
                # if simkey == 'cycleavg':
                #     TimeSeriesPlot.materializeSpikes(ax, nodedata, tplt, yplt, c, mode)

    def plotSignalsOver2DSpace(self, gridxkey, gridxvec, gridxunit, gridykey, gridyvec, gridyunit,
                               results, pltfunc, *args, yunit='', title=None, fs=10,
                               flipud=True, fliplr=False):
        ''' Plot signals over 2D space. '''
        # Create grid-like figure
        fig, axes = plt.subplots(gridxvec.size, gridyvec.size, figsize=(6, 5))

        # Re-arrange axes and labels if flipud/fliplr option is set
        supylabel_args = {}
        supxlabel_args = {'y': 1.0, 'va': 'top'}
        if flipud:
            axes = axes[::-1]
            supxlabel_args = {}
        if fliplr:
            axes = axes[:, ::-1]
            supylabel_args = {'x': 1.0, 'ha': 'right'}

        # Add title and general axes labels
        if title is not None:
            fig.suptitle(title, fontsize=fs + 2)
        fig.supxlabel(gridxkey, fontsize=fs + 2, **supxlabel_args)
        fig.supylabel(gridykey, fontsize=fs + 2, **supylabel_args)

        # Loop through the axes and plot results, while storing time ranges
        i = 0
        tranges = []
        for i, axrow in enumerate(axes):
            for j, ax in enumerate(axrow):
                hideSpines(ax)
                hideTicks(ax)
                ax.margins(0)
                if results[i, j] is not None:
                    pltfunc(ax, results[i, j], *args)
                    tranges.append(np.ptp(ax.get_xlim()))

        if len(np.unique(tranges)) > 1:
            # If more than one time range, add common x-scale to all axes
            tmin = min(tranges)
            for axrow in axes[::-1]:
                for ax in axrow:
                    trans = (ax.transData + ax.transAxes.inverted())
                    xpoints = [trans.transform([x, 0])[0] for x in [0, tmin]]
                    ax.plot(xpoints, [-0.05] * 2, c='k', lw=2, transform=ax.transAxes)
        else:
            # Otherwise, add x-scale only to axis opposite to origin
            side = 'top' if flipud else 'bottom'
            addXscale(axes[-1, -1], 0, 0.05, unit='ms', fmt='.0f', fs=fs, side=side)

        # Harmonize y-limits across all axes, and add y-scale to axis opposite to origin
        harmonizeAxesLimits(axes, dim='y')
        side = 'left' if fliplr else 'right'
        if yunit is not None:
            addYscale(axes[-1, -1], 0.05, 0, unit=yunit, fmt='.0f', fs=fs, side=side)

        # Set labels for xvec and yvec values along the two figure grid dimensions
        for ax, x in zip(axes[0, :], gridxvec):
            ax.set_xlabel(f'{si_format(x)}{gridxunit}', labelpad=15, fontsize=fs + 2)
            if not flipud:
                ax.xaxis.set_label_position('top')
        for ax, y in zip(axes[:, 0], gridyvec):
            if fliplr:
                ax.yaxis.set_label_position('right')
            ax.set_ylabel(f'{si_format(y)}{gridyunit}', labelpad=15, fontsize=fs + 2)

        # Return figure object
        return fig


class PassiveBenchmark(Benchmark):

    def __init__(self, a, nnodes, Cm0, ELeak, **kwargs):
        super().__init__(a, nnodes, **kwargs)
        self.Cm0 = Cm0
        self.ELeak = ELeak

    def pdict(self):
        return {
            **super().pdict(),
            'Cm0': f'{self.Cm0 * 1e2:.1f} uF/cm2',
            'ELeak': f'{self.ELeak} mV',
        }

    def getModelAndRunSims(self, drives, covs, taum, tauax):
        ''' Create passive model for a combination of time constants. '''
        gLeak = self.Cm0 / taum
        ga = self.Cm0 / tauax
        pneuron = passiveNeuron(self.Cm0, gLeak, self.ELeak)
        model = CoupledSonophores([
            NeuronalBilayerSonophore(self.a, pneuron) for i in range(self.nnodes)], ga)
        return self.runSims(model, drives, None, covs)

    def runSimsOverTauSpace(self, drives, covs, taum_range, tauax_range, mpi=False):
        ''' Run simulations over 2D time constant space. '''
        queue = [[drives, covs] + x for x in Batch.createQueue(taum_range, tauax_range)]
        batch = Batch(self.getModelAndRunSims, queue)
        # batch.printQueue(queue)
        output = batch.run(mpi=mpi)
        results = [x[0] for x in output]  # removing meta
        return np.reshape(results, (taum_range.size, tauax_range.size)).T

    def computeSteadyStateDivergence(self, data):
        ''' Evaluate per-node steady-state absolute deviation on charge density profiles. '''
        return {k: np.abs(np.squeeze(np.diff(self.getCharges(data, k), axis=0)))[-1]
                for k in data['cycleavg'].keys()}

    @staticmethod
    def computeAreaRatio(yref, yeval, dt):
        # Compute absolute differential signals: between reference solution and unit steady-state,
        # and between the two solutions
        signals = [np.ones_like(yref), yeval]
        diffsignals = [np.abs(y - yref) for y in signals]
        # Compute related areas
        areas = [np.sum(y) * dt for y in diffsignals]
        # Return ratio of the two areas
        ratio = areas[1] / areas[0]
        logger.debug(
            f"{', '.join([f'{x * 1e5:.2f}%.ms' for x in areas])}, ratio = {ratio * 1e2:.2f}%")
        return ratio

    def isExponentialChargeBuildup(self, Qm):
        ''' Check if charge signal corresponds to an exponential build-up. '''
        if np.ptp(Qm) < 1e-5:  # C/m2
            logger.debug('too narrow')
            return False
        Qmin, Qmax = bounds(Qm)
        Qbounds_check = dict(atol=1e-7, rtol=1e-5)
        # if not np.isclose(Qm[0], Qmin, **Qbounds_check):
        #     logger.debug('not starting from bottom')
        #     return False
        if not np.isclose(Qm[-1], Qmax, **Qbounds_check):
            logger.debug('not finishing on top')
            return False
        return True

    def computeTransientDivergence(self, data):
        ''' Evaluate per-node mean absolute difference on [0, 1] normalized charge profiles. '''
        d = {}
        t = self.getTime(data)
        dt = t[1] - t[0]
        for k in data['cycleavg'].keys():
            y = self.getCharges(data, k)
            # If cycle-avg charge profile corresponds to an exponential build-up
            if self.isExponentialChargeBuildup(y[0]):
                # Rescale signals linearly between 0 and 1
                ynorms = np.array([rescale(yy) for yy in y])
                # Restrict signals to transient phase
                tthr = self.getConvergenceTime(t, ynorms[0])
                ynorms = [yy[t <= tthr] for yy in ynorms]
                # Compute ratio between the cycle-avg steady-state convergence area and the
                # difference area between cycle-avg and sonic solutions
                d[k] = self.computeAreaRatio(*ynorms, dt) * 1e2
            else:
                d[k] = np.nan
        return d

    def eval_funcs(self):
        return {
            **super().eval_funcs(),
            'ss': (self.computeSteadyStateDivergence, 'nC/cm2', 1e5),
            'transient': (self.computeTransientDivergence, '%', 1e0)
        }

    def plotSignalsOverTauSpace(self, taum_range, tauax_range, results, pltfunc=None, fs=10):
        if pltfunc is None:
            pltfunc = 'plotQm'
        yunit = {'plotQm': 'nC/cm2', 'plotQnorm': None}[pltfunc]
        title = pltfunc[4:]
        pltfunc = getattr(self, pltfunc)
        return self.plotSignalsOver2DSpace(
            'taum', taum_range, 's', 'tauax', tauax_range, 's', results, pltfunc,
            title=title, yunit=yunit)

    @staticmethod
    def getConvergenceTime(t, y, ythr=0.999):
        i = np.where(y > ythr)[0][0]
        # return np.interp(ythr, y[i - 1:i + 1], t[i - 1:i + 1])
        return t[i]

    def plotQnorm(self, ax, data):
        t = self.getTime(data)
        for i, (k, nodedata) in enumerate(data['cycleavg'].items()):
            dt = t[1] - t[0]
            y = self.getCharges(data, k)
            c = self.nodecolors[i]
            ynorms = np.array([rescale(yy) for yy in y])
            for yn, marker in zip(ynorms, ['--', '-']):
                ax.plot(t * 1e3, yn, marker, c=c)
            ax.axhline(1., ls='--', color='k')
            if self.isExponentialChargeBuildup(y[0]):
                tthr = self.getConvergenceTime(t, ynorms[0])
                t_fill = t[t <= tthr]
                ynorms_fill = [yy[t <= tthr] for yy in ynorms]
                ax.axvline(tthr * 1e3, ls='--', color=c)
                ax.fill_between(t_fill * 1e3, *ynorms_fill, alpha=0.5, color=c)
                eps = self.computeAreaRatio(*ynorms_fill, dt)
            else:
                eps = np.nan
            ax.text(0.5, 0.3 * (i + 1), f'{eps * 1e2:.2f}%', c=c, transform=ax.transAxes)


class FiberBenchmark(Benchmark):

    def __init__(self, a, nnodes, pneuron, ga, **kwargs):
        super().__init__(a, nnodes, **kwargs)
        self.model = CoupledSonophores([
            NeuronalBilayerSonophore(self.a, pneuron) for i in range(self.nnodes)], ga)

    def pdict(self):
        return {
            **super().pdict(),
            'ga': self.model.gastr,
            'pneuron': self.model.refpneuron,
        }

    def getModelAndRunSims(self, Fdrive, tstim, covs, A1, A2):
        ''' Create passive model for a combination of time constants. '''
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A1), AcousticDrive(Fdrive, A2)])
        return self.runSims(self.model, drives, tstim, covs)

    def runSimsOverAmplitudeSpace(self, Fdrive, tstim, covs, A_range, mpi=False, subset=None):
        ''' Run simulations over 2D time constant space. '''
        # Generate 2D amplitudes meshgrid
        A_combs = np.meshgrid(A_range, A_range)
        # Set elements below main diagonal to NaN
        tril_idxs = np.tril_indices(A_range.size, -1)
        for x in A_combs:
            x[tril_idxs] = np.nan
        # Flatten the meshgrid and assemble into list of tuples
        A_combs = list(zip(*[x.flatten().tolist() for x in A_combs]))
        # Remove NaN elements
        A_combs = list(filter(lambda x: not any(np.isnan(xx) for xx in x), A_combs))
        # Assemble queue
        queue = [[Fdrive, tstim, covs] + list(x) for x in A_combs]
        # restrict queue if subset is specified
        if subset is not None:
            queue = queue[subset[0]:subset[1] + 1]
        batch = Batch(self.getModelAndRunSims, queue)
        output = batch.run(mpi=mpi)
        results = [x[0] for x in output]  # removing meta
        # Re-organize results into upper-triangle matrix
        new_results = np.empty((A_range.size, A_range.size), dtype=object)
        triu_idxs = np.triu_indices(A_range.size, 0)
        for *idx, res in zip(*triu_idxs, results):
            new_results[idx[0], idx[1]] = res
        return new_results

    def computeGamma(self, data, *args):
        ''' Evaluate per-node gamma on charge density profiles. '''
        gamma_dict = {}
        resolution = list(data['cycleavg'].values())[0].dt
        for k in data['cycleavg'].keys():
            # Get charge vectors (discarding 1st and last indexes) and compute gamma
            gamma_dict[k] = gamma(*self.getCharges(data, k), *args, resolution)
        return gamma_dict

    def plotQm(self, ax, data, *gamma_args):
        super().plotQm(ax, data)
        if len(gamma_args) > 0:
            gamma_dict = self.computeGamma(data, *gamma_args)
            tplt = self.getTime(data) * 1e3
            for i, (nodekey, nodegamma) in enumerate(gamma_dict.items()):
                # Compute states derivatives and identify transition indexes
                isgamma = nodegamma >= 1
                itransitions = np.where(np.abs(np.diff(isgamma)) > 0)[0] + 1
                if len(itransitions) > 0:
                    if isgamma[0]:
                        itransitions = np.hstack(([0], itransitions))
                    if isgamma[-1]:
                        itransitions = np.hstack((itransitions, [isgamma.size - 1]))
                    spans = list(zip(tplt[itransitions[:-1]], tplt[itransitions[1:]]))
                    for span in spans:
                        ax.axvspan(*span, ec='none', fc=self.nodecolors[i], alpha=0.2)

    def plotGamma(self, ax, data, *gamma_args):
        gamma_dict = self.computeGamma(data, *gamma_args)
        tplt = self.getTime(data) * 1e3
        for i, (nodekey, nodegamma) in enumerate(gamma_dict.items()):
            ax.plot(tplt, nodegamma, c=self.nodecolors[i], label=nodekey)
        ax.axhline(1, linestyle='--', c='k')

    def plotSignalsOverAmplitudeSpace(self, A_range, results, *args, pltfunc=None, fs=10):
        if pltfunc is None:
            pltfunc = 'plotQm'
        yunit = {'plotQm': 'nC/cm2', 'plotGamma': ''}[pltfunc]
        title = pltfunc[4:]
        pltfunc = getattr(self, pltfunc)
        return self.plotSignalsOver2DSpace(
            'A1', A_range, 'Pa', 'A2', A_range, 'Pa', results, pltfunc, *args,
            title=title, yunit=yunit)

    def computeGammaDivergence(self, data, *args):
        return {k: np.nanmax(v) for k, v in self.computeGamma(data, *args).items()}

    def eval_funcs(self):
        return {
            **super().eval_funcs(),
            'gamma': (self.computeGammaDivergence, '', 1e0)
        }
