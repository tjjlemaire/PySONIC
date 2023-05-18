# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-05-18 08:35:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-18 12:58:13

''' Main entry point for PySONIC package. '''

import logging
from .download import download_lookups
from .utils import logger

logger.setLevel(logging.INFO)

import sys

# Define available commands
commands = {
    'download': download_lookups,
}
avail_commands_str = f'Available commands: {", ".join(commands.keys())}'

# If no command is provided, print available commands and exit
if len(sys.argv) == 1:
    logger.info(avail_commands_str)
    sys.exit(0)

# Otherwise, pop the first argument
command = sys.argv.pop(1)

# If not a valid command, log error, print available commands and exit
if command not in commands.keys():
    logger.error(f'Unknown command: {command}\n{avail_commands_str}')
    sys.exit(1)

# Execute parsed command
commands[command]()

# Exit
sys.exit(0)