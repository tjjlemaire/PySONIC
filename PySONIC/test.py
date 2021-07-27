# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-28 11:55:16
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-22 15:09:38

import time
import cProfile
import pstats
import inspect
import matplotlib.pyplot as plt

from .utils import logger
from .parsers import TestParser


class TestBase:

    prefix = 'test_'
    parser_class = TestParser

    def execute(self, func, is_profiled):
        ''' Execute function with or without profiling. '''
        if is_profiled:
            profile = cProfile.Profile()
            profile.enable()
            func()
            profile.disable()
            ps = pstats.Stats(profile)
            ps.strip_dirs()
            ps.sort_stats('cumulative')
            ps.print_stats()
        else:
            func()

    def buildtestSet(self):
        ''' Build list of candidate testsets. '''
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        testsets = {}
        n = len(self.prefix)
        for name, obj in methods:
            if name[:n] == self.prefix:
                testsets[name[n:]] = obj
        return testsets

    def parseCommandLineArgs(self, testsets):
        ''' Parse command line arguments. '''
        parser = self.parser_class(list(testsets.keys()))
        parser.addHideOutput()
        args = parser.parse()
        logger.setLevel(args['loglevel'])
        if args['profile'] and args['subset'] == 'all':
            raise ValueError('profiling can only be run on individual tests')
        return args

    def runTests(self, testsets, args):
        ''' Run appropriate tests. '''
        for s in args['subset']:
            testsets[s](args['profile'])

    def main(self):
        testsets = self.buildtestSet()
        try:
            args = self.parseCommandLineArgs(testsets)
        except ValueError as err:
            logger.error(err)
            return
        t0 = time.time()
        self.runTests(testsets, args)
        tcomp = time.time() - t0
        logger.info('tests completed in %.0f s', tcomp)
        if not args['hide']:
            plt.show()
