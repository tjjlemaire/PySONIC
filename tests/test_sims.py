# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-14 18:37:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-22 15:11:50

''' Test the basic functionalities of the package. '''

from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.core.drives import AcousticDrive, ElectricDrive
from PySONIC.core.protocols import PulsedProtocol
from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron, getNeuronsDict
from PySONIC.test import TestBase


class TestSims(TestBase):

    a = 32e-9  # m
    USdrive = AcousticDrive(500e3, 100e3)

    def test_MECH(self, is_profiled=False):
        logger.info('Test: running MECH simulation')
        Qm0 = -80e-5    # membrane resting charge density (C/m2)
        Cm0 = 1e-2      # membrane resting capacitance (F/m2)
        Qm = 50e-5      # C/m2
        bls = BilayerSonophore(self.a, Cm0, Qm0)
        self.execute(lambda: bls.simulate(self.USdrive, Qm), is_profiled)

    def test_ESTIM(self, is_profiled=False):
        logger.info('Test: running ESTIM simulation')
        ELdrive = ElectricDrive(10.0)  # mA/m2
        pp = PulsedProtocol(100e-3, 50e-3)
        pneuron = getPointNeuron('RS')
        self.execute(lambda: pneuron.simulate(ELdrive, pp), is_profiled)

    def test_ASTIM_sonic(self, is_profiled=False):
        logger.info('Test: ASTIM sonic simulation')
        pp = PulsedProtocol(50e-3, 10e-3)
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(self.a, pneuron)

        # test error 1: sonophore radius outside of lookup range
        try:
            nbls = NeuronalBilayerSonophore(100e-9, pneuron)
            nbls.simulate(self.USdrive, pp, method='sonic')
        except ValueError:
            logger.debug('Out of range radius: OK')

        # test error 2: frequency outside of lookups range
        try:
            nbls = NeuronalBilayerSonophore(self.a, pneuron)
            nbls.simulate(AcousticDrive(10e3, self.USdrive.A), pp, method='sonic')
        except ValueError:
            logger.debug('Out of range frequency: OK')

        # test error 3: amplitude outside of lookups range
        try:
            nbls = NeuronalBilayerSonophore(self.a, pneuron)
            nbls.simulate(AcousticDrive(self.USdrive.f, 1e6), pp, method='sonic')
        except ValueError:
            logger.debug('Out of range amplitude: OK')

        # Run simulation on all neurons
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR', 'SWnode'):
                pneuron = neuron_class()
                nbls = NeuronalBilayerSonophore(self.a, pneuron)
                self.execute(lambda: nbls.simulate(self.USdrive, pp, method='sonic'), is_profiled)

    def test_ASTIM_full(self, is_profiled=False):
        logger.info('Test: running ASTIM detailed simulation')
        pp = PulsedProtocol(1e-6, 1e-6)
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(self.a, pneuron)
        self.execute(lambda: nbls.simulate(self.USdrive, pp, method='full'), is_profiled)

    def test_ASTIM_hybrid(self, is_profiled=False):
        logger.info('Test: running ASTIM hybrid simulation')
        pp = PulsedProtocol(0.6e-3, 0.1e-3)
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(self.a, pneuron)
        self.execute(lambda: nbls.simulate(self.USdrive, pp, method='hybrid'), is_profiled)


if __name__ == '__main__':
    tester = TestSims()
    tester.main()
