import json
import logging
import logging.config
from multiprocessing import Manager
import os
import pathlib
import sys
import platform
from OpenGL.plugins import FormatHandler
from PyQt6 import QtCore
from sscanss.__version import __version__
from sscanss.settings import Setting

if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ['QT_QPA_PLATFORM'] = 'wayland-egl'

if getattr(sys, "frozen", False):
    # we are running in a bundle
    SOURCE_PATH = pathlib.Path(sys.executable).parent.parent
    if pathlib.Path(SOURCE_PATH / 'MacOS').is_dir():
        SOURCE_PATH = SOURCE_PATH / 'Resources'
    from sscanss.__config_data import LOG_CONFIG, SCHEMA
    INSTRUMENT_SCHEMA = SCHEMA
else:
    SOURCE_PATH = pathlib.Path(__file__).parent.parent
    with open(SOURCE_PATH / "logging.json", "r") as log_file:
        LOG_CONFIG = json.loads(log_file.read())

    with open(SOURCE_PATH / "instrument_schema.json", "r") as schema_file:
        INSTRUMENT_SCHEMA = json.loads(schema_file.read())

DOCS_URL = f'https://isisneutronmuon.github.io/SScanSS-2/{__version__}/index.html'
UPDATE_URL = 'https://api.github.com/repos/ISISNeutronMuon/SScanSS-2/releases/latest'
RELEASES_URL = 'https://github.com/ISISNeutronMuon/SScanSS-2/releases'
INSTRUMENTS_PATH = SOURCE_PATH / 'instruments'

# Tells OpenGL to use the NumpyHandler for the Matrix44 objects
FormatHandler('sscanss', 'OpenGL.arrays.numpymodule.NumpyHandler', ['sscanss.core.math.matrix.Matrix44'])


def handle_scaling():
    """Changes settings to handle UI scaling"""
    os_type = platform.system()
    if os_type == "Windows":
        from ctypes import windll
        windll.user32.SetProcessDPIAware()


def set_locale():
    locale = QtCore.QLocale(QtCore.QLocale.Language.C)
    locale.setNumberOptions(QtCore.QLocale.NumberOption.RejectGroupSeparator)
    QtCore.QLocale.setDefault(locale)


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """
    Qt slots swallows exceptions but this ensures exceptions are logged
    """
    logging.error('An unhandled exception occurred!', exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()
    sys.exit(1)


def setup_logging(filename):
    """
    Configure of logging file handler.

    :param filename: name of log file
    :type filename: str
    """
    try:
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        LOG_CONFIG['handlers']['file_handler']['filename'] = LOG_PATH / filename
        logging.config.dictConfig(LOG_CONFIG)
    except OSError:
        logging.basicConfig(level=logging.ERROR)
        logging.exception('Could not initialize logging to file')

    sys.excepthook = log_uncaught_exceptions


class ProcessServer:
    """Singleton which holds a multiprocessing manager. This speeds up simulation
    by creating the manager on startup (not before the simulation) and caching the manager"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = Manager()
        return cls._instance


set_locale()
settings = Setting()
LOG_PATH = pathlib.Path(settings.filename()).parent / 'logs'
