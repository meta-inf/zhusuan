# -*- coding: utf-8 -*-
# File: logger.py
# Adapted from Tensorpack


import logging
import os
import shutil
import os.path
from termcolor import colored
from datetime import datetime
from six.moves import input
import sys

__all__ = ['set_logger_dir', 'auto_set_dir', 'get_logger_dir']


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'green') + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


def _getlogger():
    logger = logging.getLogger('zhusuan')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug', 'setLevel']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)

