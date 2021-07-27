# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-07-30 15:33:22
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-08 15:46:35

import logging
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from PySONIC.core import BilayerSonophore, AcousticDrive
from PySONIC.utils import logger
from PySONIC.postpro import filtfilt, computeTimeStep
from PySONIC.constants import NPC_DENSE

logger.setLevel(logging.INFO)

# Constants
MAX_PROFILES = 6       # max number of profiles to display simultaneously on figure


def invfiltfilt(y, *args, **kwargs):
    ''' Inverse signal before and after filtering. '''
    return 1 / filtfilt(1 / y, *args, **kwargs)


def getCmProfiles(bls, drive, nreps):
    ''' Simulate mechanical model with a specific drive and return extended
        time, capacitance and capacitance sinusoidal approximation profiles.
    '''
    data, _ = bls.simulate(drive, bls.Qm0)
    logger.info('Extracting detailed capacitance profile')
    Z_last = data.tail(NPC_DENSE)['Z'].values  # m
    Cm_last = bls.v_capacitance(Z_last)  # F/m2
    Cm = np.tile(Cm_last, nreps)
    t = np.linspace(0, nreps / drive.f, Cm.size)
    gamma = np.ptp(Cm) / (2 * bls.Cm0)
    logger.info(f'Generating corresponding pure sinusoid capacitance profile (gamma = {gamma:.2f})')
    Cm_approx = bls.Cm0 * (1 + gamma * np.sin(2 * np.pi * f_US * t))  # F/m2
    return t, Cm, Cm_approx


def getSecondHalfAvg(x):
    ''' Extract the effective capacitance from the second half of a capacitance profile. '''
    return np.squeeze(np.nanmean(x[x.shape[0] // 2:], axis=0))


def plotRelCmfiltsVsCutoff(rel_fcs, rel_Cm, rel_Cmfilts, condition):
    ''' Plot an original Cm profile and filtered variants at various cutoff frequencies. '''
    rsf = int(np.ceil(rel_fcs.size / MAX_PROFILES))  # potential resampling factor
    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'Cm profiles vs. cutoff ({condition})')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('Cm / Cm0')
    ax.plot(t * 1e6, rel_Cm, label='unfiltered', c='k')
    ax.axhline(np.mean(rel_Cm), c='k', linestyle='--')
    ax.axhline(1 / np.mean(1 / rel_Cm), c='k', linestyle=':')
    for i, (rel_fc, rel_Cmfilt) in enumerate(zip(rel_fcs[::rsf], rel_Cmfilts[::rsf])):
        ax.plot(t * 1e6, rel_Cmfilt, label=f'$f_c = {rel_fc:.2g}\\ f_{{US}}$', c=colors[i])
        ax.axhline(getSecondHalfAvg(rel_Cmfilt), c=colors[i], linestyle='--')
    ax.legend()
    fig.tight_layout()
    return fig


def plotRelCmeffVsCutoff(rel_fcs, rel_Cmavgs, rel_Cmeffs, rel_Cmfilts, condition, colors=None):
    ''' Plot effective capacitance as a function of cutoff frequency for various conditions. '''
    fig, ax = plt.subplots()
    if colors is None:
        colors = plt.get_cmap('tab10').colors
    ax.set_title(f'Cmeff vs. cutoff - {condition}')
    ax.set_xlabel('$f_c / f_{US}$')
    ax.set_ylabel('$C_{m, eff} / C_{m0}$')
    ax.set_xscale('log')
    for (k, Cm), c in zip(rel_Cmfilts.items(), colors):
        ax.plot(rel_fcs, getSecondHalfAvg(Cm.T), label=k, c=c)
        ax.axhline(rel_Cmavgs[k], linestyle='--', c=c)
        ax.axhline(rel_Cmeffs[k], linestyle=':', c=c)
    # if gamma is not None:
    #     ax.axhline(np.sqrt(1 - gamma**2), c='k', linestyle='--', label='$\\sqrt{1 - \\gamma^2}$')
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot profiles')
    args = ap.parse_args()

    # Mechanical model
    a = 32e-9     # m
    Cm0 = 1e-2    # resting capacitance (F/m2)
    Qm0 = 0.0     # resting charge density (C/m2)
    bls = BilayerSonophore(a, Cm0, Qm0)

    # Acoustic parameters
    freqs = np.array([20., 500., 4000.]) * 1e3     # Hz
    amps = np.logspace(1, 3, 3) * 1e3  # Pa

    # Define colors
    colors = list(plt.get_cmap('tab20c').colors)
    del colors[3::4]
    amps = amps[::-1]

    # Filter parameters
    order = 2                          # filter order
    rel_fcs = np.logspace(-1, 3, 100)  # relative cutoff frequencies w.r.t. US frequency
    nreps = int(2 / rel_fcs.min())     # minimum number of acoustic cycles

    # Plot parameters
    plot_profiles = args.plot

    variants = ['detailed', 'approx']
    rel_Cmavgs = {k: {} for k in variants}
    rel_Cmeffs = {k: {} for k in variants}
    rel_Cmfilts = {k: {} for k in variants}

    for f_US in freqs:
        fcs = rel_fcs * f_US
        for A_US in amps:
            drive = AcousticDrive(f_US, A_US)
            label = drive.desc

            # Get original Cm signal and sinusoidal approximation
            t, *Cms = getCmProfiles(bls, drive, nreps)

            # Get sampling and Nyquist frequency from time signal
            fs = 1 / computeTimeStep(t)
            fnyq = fs / 2

            # Warn if Nyquist frequency is lower than max cutoff frequency
            if fcs.max() > fnyq:
                logger.warning(
                    f'max cutoff {fcs.max() / fnyq:.2f} times higher than signal Nyquist')

            # For each Cm profile variant
            for k, Cm in zip(variants, Cms):
                # Normalize by resting capacitance
                rel_Cm = Cm / Cm0

                # Compute average and effective metrics
                rel_Cmavgs[k][label] = rel_Cm.mean()
                rel_Cmeffs[k][label] = 1 / np.mean(1 / rel_Cm)

                # Filter reciprocal at various cutoff frequencies (except those above nyquist)
                rel_Cmfilts_list = []
                for fc in fcs[fcs <= fnyq]:
                    rel_Cmfilts_list.append(invfiltfilt(rel_Cm, fs, fc, order))
                for fc in fcs[fcs > fnyq]:
                    rel_Cmfilts_list.append(np.ones(rel_Cm.size) * np.nan)
                rel_Cmfilts[k][label] = np.array(rel_Cmfilts_list)

                if plot_profiles:
                    # Plot filtered profiles for a subset of cutoff frequencies
                    plotRelCmfiltsVsCutoff(rel_fcs, rel_Cm, rel_Cmfilts[k][label], label)

    for k in variants:
        # Plot effective capacitance as a function of cutoff frequency for various conditions
        plotRelCmeffVsCutoff(
            rel_fcs, rel_Cmavgs[k], rel_Cmeffs[k], rel_Cmfilts[k], k, colors=colors)

    plt.show()
