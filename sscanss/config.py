from contextlib import suppress
from enum import Enum, unique
import json
import logging
import logging.config
from multiprocessing import Manager
import pathlib
import sys
import platform
from OpenGL.plugins import FormatHandler
from PyQt5 import QtCore, QtWidgets
from sscanss.__version import __version__

if getattr(sys, "frozen", False):
    # we are running in a bundle
    SOURCE_PATH = pathlib.Path(sys.executable).parent.parent
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
CUSTOM_INSTRUMENTS_PATH = pathlib.Path.home() / 'Documents' / 'SScanSS 2' / 'instruments'
STATIC_PATH = SOURCE_PATH / 'static'
IMAGES_PATH = STATIC_PATH / 'images'

# Tells OpenGL to use the NumpyHandler for the Matrix44 objects
FormatHandler('sscanss', 'OpenGL.arrays.numpymodule.NumpyHandler', ['sscanss.core.math.matrix.Matrix44'])


def path_for(filename):
    """Gets full path for the given image file

    :param filename: basename and extension of image
    :type filename: str
    :return: full path of image
    :rtype: str
    """
    return (IMAGES_PATH / filename).as_posix()


def load_stylesheet(name):
    """Loads qt stylesheet from file

    :param name: path to stylesheet file
    :type name: str
    :return: stylesheet
    :rtype: str
    """
    with suppress(FileNotFoundError):
        with open(STATIC_PATH / name, 'rt') as stylesheet:
            style = stylesheet.read().replace('@Path', IMAGES_PATH.as_posix())
        return style
    return ''


def handle_scaling():
    """Changes settings to handle UI scaling"""
    os_type = platform.system()
    if os_type == "Windows":
        from ctypes import windll
        windll.user32.SetProcessDPIAware()
    else:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


@unique
class Group(Enum):
    """Setting groups"""
    General = 'General'
    Graphics = 'Graphics'
    Simulation = 'Simulation'


@unique
class Key(Enum):
    """Setting keys"""
    Geometry = 'Geometry'
    Check_Update = 'Check_Update'
    Recent_Projects = 'Recent_Projects'
    Recent_Editor_Projects = 'Recent_Editor_Projects'
    Align_First = f'{Group.Simulation.value}/Align_First'
    Position_Stop_Val = f'{Group.Simulation.value}/Position_Stop_Val'
    Angular_Stop_Val = f'{Group.Simulation.value}/Angular_Stop_Val'
    Local_Max_Eval = f'{Group.Simulation.value}/Local_Max_Eval'
    Global_Max_Eval = f'{Group.Simulation.value}/Global_Max_Eval'
    Skip_Zero_Vectors = f'{Group.Simulation.value}/Skip_Zero_Vectors'
    Sample_Colour = f'{Group.Graphics.value}/Sample_Colour'
    Fiducial_Colour = f'{Group.Graphics.value}/Fiducial_Colour'
    Fiducial_Disabled_Colour = f'{Group.Graphics.value}/Fiducial_Disabled_Colour'
    Measurement_Colour = f'{Group.Graphics.value}/Measurement_Colour'
    Measurement_Disabled_Colour = f'{Group.Graphics.value}/Measurement_Disabled_Colour'
    Vector_1_Colour = f'{Group.Graphics.value}/Vector_1_Colour'
    Vector_2_Colour = f'{Group.Graphics.value}/Vector_2_Colour'
    Selected_Colour = f'{Group.Graphics.value}/Selected_Colour'
    Cross_Sectional_Plane_Colour = f'{Group.Graphics.value}/Cross_Sectional_Plane_Colour'
    Fiducial_Size = f'{Group.Graphics.value}/Fiducial_Size'
    Measurement_Size = f'{Group.Graphics.value}/Measurement_Size'
    Vector_Size = f'{Group.Graphics.value}/Vector_Size'
    Custom_Instruments_Path = f'{Group.General.value}/Custom_Instruments_Path'


class SettingItem:
    """Creates a setting item

    :param default: default value for the setting item
    :type default: Any
    :param limits: lower and upper bounds of item
    :type limits: Union[(Any, Any), None]
    :param sub_type: type of the contents of iterable items
    :type sub_type: type object
    :param fixed_size: size of iterable item
    :type fixed_size: int
    """
    def __init__(self, default, limits=None, sub_type=None, fixed_size=False):
        self.default = default
        self.type = type(default)
        self.sub_type = sub_type
        self.size = 0
        with suppress(TypeError):
            self.size = len(default)
        self.fixed_size = fixed_size
        self.limits = limits


__defaults__ = {
    Key.Geometry: SettingItem(bytearray(b'')),
    Key.Check_Update: SettingItem(True),
    Key.Skip_Zero_Vectors: SettingItem(False),
    Key.Align_First: SettingItem(True),
    Key.Recent_Projects: SettingItem([], sub_type=str),
    Key.Recent_Editor_Projects: SettingItem([], sub_type=str),
    Key.Local_Max_Eval: SettingItem(1000, limits=(500, 5000)),
    Key.Global_Max_Eval: SettingItem(200, limits=(50, 500)),
    Key.Angular_Stop_Val: SettingItem(1.00, limits=(0.000, 360.000)),
    Key.Position_Stop_Val: SettingItem(1e-2, limits=(0.000, 100.000)),
    Key.Custom_Instruments_Path: SettingItem(str(CUSTOM_INSTRUMENTS_PATH)),
    Key.Sample_Colour: SettingItem((0.65, 0.65, 0.65, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Fiducial_Colour: SettingItem((0.4, 0.9, 0.4, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Fiducial_Disabled_Colour: SettingItem((0.9, 0.4, 0.4, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Measurement_Colour: SettingItem((0.01, 0.44, 0.12, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Measurement_Disabled_Colour: SettingItem((0.9, 0.4, 0.4, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Vector_1_Colour: SettingItem((0.0, 0.0, 1.0, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Vector_2_Colour: SettingItem((1.0, 0.0, 0.0, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Selected_Colour: SettingItem((0.94, 0.82, 0.68, 1.0), sub_type=float, limits=(0.0, 1.0), fixed_size=4),
    Key.Cross_Sectional_Plane_Colour: SettingItem((0.93, 0.83, 0.53, 1.0),
                                                  sub_type=float,
                                                  limits=(0.0, 1.0),
                                                  fixed_size=4),
    Key.Fiducial_Size: SettingItem(5, limits=(5, 30)),
    Key.Measurement_Size: SettingItem(5, limits=(5, 30)),
    Key.Vector_Size: SettingItem(10, limits=(10, 50))
}


class Setting:
    """Class handles storage and retrieval of application settings as Key-Value pairs.
    A key could belong to a group e.g Graphics (Graphics/Colour) or be generic like the
    Geometry setting. The setting are written to a .INI file.
    """
    def __init__(self):
        self.local = {}
        self.system = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, 'SScanSS 2', 'SScanSS 2')

    @staticmethod
    def default(key):
        return __defaults__[key]

    def value(self, key):
        """Retrieves the value saved with the given key or the default value if no value is
        saved.

        :param key: setting key
        :type key: Enum
        :return: value saved with given key or default
        :rtype: Any
        """
        item = self.default(key)
        return self.__getSafeValue(key, item)

    def __getSafeValue(self, key, item):
        """Retrieves the safe value of given key. if the value is not safe (e.g. outside bounds, wrong type etc)
        the default value is returned

        :param key: setting key
        :type key: Enum
        :param item: default item
        :type item: SettingItem
        :return: value saved with given key or default
        :rtype: Any
        """
        try:
            if key.value in self.local:
                value = self.local[key.value]
            else:
                value = self.system.value(key.value, item.default)

            if item.type is int:
                value = int(value)
                if item.limits is not None and (value > item.limits[1] or value < item.limits[0]):
                    return item.default
                return value

            elif item.type is float:
                value = float(value)
                if item.limits is not None and (value > item.limits[1] or value < item.limits[0]):
                    return item.default
                return value

            elif item.type is bool:
                if type(value) is bool:
                    return value
                else:
                    # QSetting stores boolean as string in ini file
                    return (value.lower() == 'true') if type(value) is str else item.default

            elif item.type is list or item.type is tuple:
                if type(value) is str:
                    # QSetting could return string when list contains single value
                    value = [value] if value else item.default

                if item.sub_type is not None:
                    value = item.type(map(item.sub_type, value))

                if item.fixed_size and item.size != len(value):
                    return item.default

                if item.limits is not None:
                    for v in value:
                        if v > item.limits[1] or v < item.limits[0]:
                            return item.default

                return value

            else:
                return item.type(value)

        except (ValueError, TypeError):
            return item.default

    def setValue(self, key, value, default=False):
        """Set value of a setting key

        :param key: setting key
        :type key: Enum
        :param value: new value
        :type value: Any
        :param default: indicates the system setting should also be set
        :type default: bool
        """
        self.local[key.value] = value
        if default:
            self.system.setValue(key.value, value)

    def reset(self, default=False):
        """Clear saved values of setting keys that belong to a Group. Keys without
        a group e.g. Check_Update are not cleared.

        :param default: indicates the system setting should also be reset
        :type default: bool
        """
        self.local.clear()
        if default:
            for group in self.Group:
                self.system.remove(group.value)

    def filename(self):
        """Returns full path of setting file

        :return: setting file path
        :rtype: str
        """
        return self.system.fileName()


def set_locale():
    locale = QtCore.QLocale(QtCore.QLocale.C)
    locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)
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
setattr(settings, 'Key', Key)
setattr(settings, 'Group', Group)
LOG_PATH = pathlib.Path(settings.filename()).parent / 'logs'
