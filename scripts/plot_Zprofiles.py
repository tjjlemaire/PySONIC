# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-08-17 15:29:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-08-17 19:24:19

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import Batch, BilayerSonophore, AcousticDrive
from PySONIC.utils import logger

logger.setLevel(logging.INFO)


def plotZProfiles(bls, drive, Qrange, mpi=False, ax=None):
    queue = bls.simQueue([drive.f], [drive.A], Qrange)
    batch = Batch(bls.getZlast, queue)
    outputs = batch(mpi=mpi)
    Zprofiles = np.array(outputs)
    t = np.linspace(0., 1. / f, Zprofiles.shape[1])
    add_legend = False
    if ax is None:
        fig, ax = plt.subplots()
        add_legend = True
    ax.set_title(drive.desc)
    ax.set_xlabel('t (us)')
    ax.set_ylabel('Z (nm)')
    handles = []
    for Z, Q in zip(Zprofiles, Qrange):
        handles.append(ax.plot(t * 1e6, Z * 1e9, label=f'Qm = {Q * 1e5:.0f} nC/cm2'))
    if add_legend:
        ax.legend(loc=1, frameon=False)
    else:
        return handles


if __name__ == '__main__':

    # Model
    a = 32e-9   # m
    Cm0 = 1e-2  # F/m2
    Qm0 = 0.    # C/m2
    bls = BilayerSonophore(a, Cm0, Qm0)

    # Stimulation parameters
    freqs = np.array([20., 100., 500., 2500.]) * 1e3   # Hz
    amps = np.array([10., 50., 100., 500., 1000.]) * 1e3  # Pa

    # Charges
    Qrange = np.linspace(0., 100., 6) * 1e-5  # C/m2

    # Sims and plots
    fig, axes = plt.subplots(freqs.size, amps.size)
    for i, f in enumerate(freqs):
        for j, A in enumerate(amps):
            handles = plotZProfiles(bls, AcousticDrive(f, A), Qrange, ax=axes[i, j])

    plt.show()
