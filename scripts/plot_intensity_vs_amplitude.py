# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-13 18:04:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-14 11:48:21

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import Pressure2Intensity, Intensity2Pressure


def plotIntensityVsAmplitude(rho, c, fs=12):
	''' Plot acoustic intenstiy (W/cm2) as a function of acoustic peak pressure amplitude (kPa).

		:param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: figure handle
	'''
	# Determine acoustic impedance
	Z = rho * c  # kg.m-2.s-1

	# Define intensity range and compute corresponding amplitudes
	I_plot = np.logspace(-3, 3, 100)  # W/cm2
	A_plot = Intensity2Pressure(I_plot * 1e4, rho=rho, c=c) * 1e-3 # kPa

	# Define characteristic intensities and amplitudes
	I_marks = np.logspace(-2, 2, 5)  # W/cm2
	A_marks = np.logspace(1, 3, 3)  # kPa

	# Create figure
	fig, ax = plt.subplots()
	ax.set_title(f'Z = {Z:.2e} kg/m2/s', fontsize=fs)
	ax.set_xlabel('Pressure amplitude (kPa)', fontsize=fs)
	ax.set_ylabel('Acoustic Intensity (W/cm2)', fontsize=fs)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(A_plot.min(), A_plot.max())
	ax.set_ylim(I_plot.min(), I_plot.max())
	for item in ax.get_xticklabels() + ax.get_yticklabels():
		item.set_fontsize(fs)

	# Plot P-I profile along with correspondance indicators for characteristic values
	ax.plot(A_plot, I_plot, c='C0')
	for I in I_marks:
		A = Intensity2Pressure(I * 1e4, rho=rho, c=c) * 1e-3 # kPa
		ax.plot([A] * 2, [I_plot.min(), I], '--', c='k')
		ax.plot([A_plot.min(), A], [I] * 2, '--', c='k')
	for A in A_marks:
		I = Pressure2Intensity(A * 1e3, rho=rho, c=c) * 1e-4  # W/cm2
		ax.plot([A] * 2, [I_plot.min(), I], '-.', c='k')
		ax.plot([A_plot.min(), A], [I] * 2, '-.', c='k')

	return fig


def main():

	parser = ArgumentParser()
	parser.add_argument('--rho', type=float, default=1075., help='Medium density (kg/m3)')
	parser.add_argument('-c', type=float, default=1515., help='Medium speed of sound (m/s)')
	args = parser.parse_args()

	fig = plotIntensityVsAmplitude(args.rho, args.c)
	plt.show()


if __name__ == '__main__':
	main()