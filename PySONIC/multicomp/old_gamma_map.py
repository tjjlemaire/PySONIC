# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-24 19:00:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-25 15:13:38

import os
import numpy as np
import matplotlib.pyplot as plt

from ..core import AcousticDrive, Lookup
from ..utils import logger, si_format, bounds, rangecode
from ..plt import XYMap
from ..constants import NPC_DENSE


class GammaMap(XYMap):
    ''' Interface to a 2D map showing relative capacitance oscillation amplitude
        resulting from BLS simulations at various frequencies and amplitude.
    '''
    xkey = 'f'
    xfactor = 1e0
    xunit = 'Hz'
    ykey = 'A'
    yfactor = 1e0
    yunit = 'Pa'
    zkey = 'gamma'
    zfactor = 1e0
    zunit = '-'
    suffix = 'gamma'

    def __init__(self, root, bls, Qm, freqs, amps):
        self.bls = bls.copy()
        self.Qm = Qm
        super().__init__(root, freqs, amps)

    @property
    def title(self):
        return f'Gamma map - {self.bls}'

    @property
    def pdict(self):
        return {
            'a': f'{self.bls.a * 1e9:.0f}nm',
            'Cm0': f'{self.bls.Cm0 * 1e2:.1f}uF_cm2',
            'Qm0': f'{self.bls.Qm0 * 1e5:.0f}nC_cm2',
            'Qm': f'{self.Qm * 1e5:.0f}nC_cm2',
        }

    @property
    def pcode(self):
        return 'bls_' + '_'.join([f'{k}{v}' for k, v in self.pdict.items()])

    def corecode(self):
        return f'gamma_map_{self.pcode}'

    def compute(self, x):
        f, A = x
        data = self.bls.simCycles(AcousticDrive(f, A), self.Qm).tail(NPC_DENSE)
        Cm = self.bls.v_capacitance(data['Z'])
        gamma = np.ptp(Cm) / self.bls.Cm0
        logger.info(f'f = {si_format(f, 1)}Hz, A = {si_format(A)}Pa, gamma = {gamma:.2f}')
        return gamma

    def onClick(self, event):
        ''' Show capacitance profile when the user clicks on a cell in the 2D map. '''
        x = self.getOnClickXY(event)
        f, A = x

        # Simulate mechanical model
        data = self.bls.simCycles(AcousticDrive(f, A), self.Qm).tail(NPC_DENSE)

        # Retrieve time and relative capacitance profiles from last cycle
        t = data['t'].values
        rel_Cm = self.bls.v_capacitance(data['Z']) / self.bls.Cm0

        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(f'f = {si_format(f, 1)}Hz, A = {si_format(A)}Pa')
        ax.set_xlabel('time (us)')
        ax.set_ylabel('Cm / Cm0')
        for sk in ['right', 'top']:
            ax.spines[sk].set_visible(False)

        # Plot capacitance profile
        ax.plot((t - t[0]) * 1e6, rel_Cm)
        ax.axhline(1.0, c='k', linewidth=0.5)

        # Indicate relative oscillation range
        ybounds = bounds(rel_Cm)
        gamma = ybounds[1] - ybounds[0]
        for y in ybounds:
            ax.axhline(y, linestyle='--', c='k')
        axis_to_data = ax.transAxes + ax.transData.inverted()
        data_to_axis = axis_to_data.inverted()
        ax_ybounds = [data_to_axis.transform((ax.get_ylim()[0], y))[1] for y in ybounds]
        xarrow = 0.9
        ax.text(0.85, np.mean(ax_ybounds), f'gamma = {gamma:.2f}', transform=ax.transAxes,
                rotation='vertical', va='center', ha='center', color='k', fontsize=10)
        ax.annotate(
            '', xy=(xarrow, ax_ybounds[0]), xytext=(xarrow, ax_ybounds[1]),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='k', edgecolor='k', arrowstyle='<|-|>'))

        # Show figure
        plt.show()

    def render(self, xscale='log', yscale='log', figsize=(6, 4), **kwargs):
        ''' Render with specific log scale. '''
        return super().render(xscale=xscale, yscale=yscale, figsize=figsize, **kwargs)

    def toPickle(self, root):
        ''' Ouput map to a lookup file (adding amplitude-zero). '''
        lkp = Lookup(
            {'f': self.xvec, 'A': np.hstack(([0.], self.yvec))},
            {'gamma': np.vstack([np.zeros(self.xvec.size), self.getOutput()]).T})

        xcode = rangecode(lkp.refs['f'], self.xkey, self.xunit)
        ycode = rangecode(lkp.refs['A'], self.ykey, self.yunit)
        xycode = '_'.join([xcode, ycode])
        lkp.toPickle(os.path.join(root, f'gamma_lkp_{self.pcode}_{xycode}.lkp'))
