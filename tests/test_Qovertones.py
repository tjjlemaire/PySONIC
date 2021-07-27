# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-03-19 19:01:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-03-23 20:43:55

import itertools
import logging
import numpy as np

from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive, Batch
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.constants import *

logger.setLevel(logging.DEBUG)

# Model
a = 32e-9
fs = 1.
pneuron = getPointNeuron('RS')
nbls = NeuronalBilayerSonophore(a, pneuron)

# Stimulus
Fdrive = 500e3
Adrive = 100e3
drive = AcousticDrive(Fdrive, Adrive)

# Charge range
Qmin, Qmax = pneuron.Qbounds
Q_ref = np.arange(Qmin, Qmax + DQ_LOOKUP, DQ_LOOKUP)  # C/m2

# Charge oscillations
AQ_ref = np.linspace(0, 100e-5, 5)  # C/m2
phiQ_ref = np.linspace(0, 2 * np.pi, 5, endpoint=False)  # rad
NFS = 2
Qovertones_dims = []
for i in range(NFS):
    Qovertones_dims += [AQ_ref, phiQ_ref]
Qovertones = Batch.createQueue(*Qovertones_dims)
Qovertones = [list(zip(x, x[1:]))[::2] for x in Qovertones]

queue = []
for Q in Q_ref:
    for Qov in Qovertones:
        queue.append([drive, fs, Q, Qov])

mpi = False
loglevel = logger.getEffectiveLevel()
batch = Batch(nbls.computeEffVars, queue)
outputs = batch(mpi=mpi, loglevel=loglevel)

# Split comp times and effvars from outputs
effvars, tcomps = [list(x) for x in zip(*outputs)]
effvars = list(itertools.chain.from_iterable(effvars))

print(effvars)
