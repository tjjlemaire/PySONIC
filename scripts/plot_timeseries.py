# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 12:41:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 20:16:17

''' Plot temporal profiles of specific simulation output variables. '''

import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import CompTimeSeries, GroupedTimeSeries
from PySONIC.parsers import TimeSeriesParser


def main():
    # Parse command line arguments
    parser = TimeSeriesParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Plot appropriate graph
    if args['compare']:
        if args['plot'] == ['all'] or args['plot'] is None:
            logger.error('Specific variables must be specified for comparative plots')
            return
        for pltvar in args['plot']:
            try:
                comp_plot = CompTimeSeries(args['inputfiles'], pltvar)
                comp_plot.render(
                    patches=args['patches'],
                    spikes=args['spikes'],
                    frequency=args['sr'],
                    trange=args['trange'],
                    prettify=args['pretty'],
                    cmap=args['cmap'],
                    cscale=args['cscale']
                )
            except KeyError as e:
                logger.error(e)
                return
    else:
        scheme_plot = GroupedTimeSeries(args['inputfiles'], pltscheme=args['pltscheme'])
        scheme_plot.render(
            patches=args['patches'],
            spikes=args['spikes'],
            frequency=args['sr'],
            trange=args['trange'],
            prettify=args['pretty'],
            save=args['save'],
            outputdir=args['outputdir'],
            fig_ext=args['figext']
        )

    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()
