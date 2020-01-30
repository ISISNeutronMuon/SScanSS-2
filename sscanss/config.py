from enum import Enum, unique
import json
import logging
import logging.config
import pathlib
import sys
from PyQt5 import QtCore

__version__ = '1.0.0-beta'

if getattr(sys, 'frozen', False):
    # we are running in a bundle
    SOURCE_PATH = pathlib.Path(sys.executable).parent.parent
else:
    SOURCE_PATH = pathlib.Path(__file__).parent.parent

DOCS_URL = 'https://isisneutronmuon.github.io/SScanSS-2/'
INSTRUMENTS_PATH = SOURCE_PATH / 'instruments'
CUSTOM_INSTRUMENTS_PATH = pathlib.Path.home() / 'Documents' / 'SScanSS-2' / 'instruments'
STATIC_PATH = SOURCE_PATH / 'static'
IMAGES_PATH = STATIC_PATH / 'images'
LOG_CONFIG_PATH = SOURCE_PATH / 'logging.json'


def path_for(filename):
    return str((IMAGES_PATH / filename).as_posix())


@unique
class Group(Enum):
    General = 'General'
    Graphics = 'Graphics'
    Simulation = 'Simulation'


@unique
class Key(Enum):
    Geometry = 'Geometry'
    Check_Update = 'Check_Update'
    Recent_Projects = 'Recent_Projects'
    Align_First = f'{Group.Simulation.value}/Align_First'
    Position_Stop_Val = f'{Group.Simulation.value}/Position_Stop_Val'
    Angular_Stop_Val = f'{Group.Simulation.value}/Angular_Stop_Val'
    Local_Max_Eval = f'{Group.Simulation.value}/Local_Max_Eval'
    Global_Max_Eval = f'{Group.Simulation.value}/Global_Max_Eval'
    Sample_Colour = f'{Group.Graphics.value}/Sample_Enabled_Colour'
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


__defaults__ = {Key.Geometry: bytearray(b''), Key.Check_Update: True, Key.Recent_Projects: [],
                Key.Local_Max_Eval: 1000, Key.Global_Max_Eval: 200, Key.Align_First: True,
                Key.Angular_Stop_Val: 1.00, Key.Position_Stop_Val: 1e-2,
                Key.Sample_Colour: (0.65, 0.65, 0.65, 1.0), Key.Custom_Instruments_Path: str(CUSTOM_INSTRUMENTS_PATH),
                Key.Fiducial_Colour: (0.4, 0.9, 0.4, 1.0), Key.Fiducial_Disabled_Colour: (0.9, 0.4, 0.4, 1.0),
                Key.Measurement_Colour: (0.01, 0.44, 0.12, 1.0), Key.Measurement_Disabled_Colour: (0.9, 0.4, 0.4, 1.0),
                Key.Vector_1_Colour: (0.0, 0.0, 1.0, 1.0), Key.Vector_2_Colour: (1.0, 0.0, 0.0, 1.0),
                Key.Selected_Colour: (0.94, 0.82, 0.68, 1.0), Key.Cross_Sectional_Plane_Colour: (0.93, 0.83, 0.53, 1.0),
                Key.Fiducial_Size: 5, Key.Measurement_Size: 5, Key.Vector_Size: 10}


class Setting:
    """Class handles storage and retrieval of application settings as Key-Value pairs.
    A key could belong to a group e.g Graphics (Graphics/Colour) or be generic like the
    Geometry setting. The setting are written to a .INI file.
    """
    Key = Key
    Group = Group

    def __init__(self):
        self.local = {}
        self.system = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                                       'SScanSS 2', 'SScanSS 2')

    def value(self, key):
        """Retrieves the value saved with the given key or the default value if no value is
        saved.

        :param key: setting key
        :type key: Enum
        :return: value saved with given key or default
        :rtype: Any
        """
        default = __defaults__[key]
        if key.value in self.local:
            value = self.local[key.value]
        else:
            value = self.system.value(key.value, default)

        if type(default) is int:
            return int(value)
        if type(default) is float:
            return float(value)
        if type(default) is bool and type(value) is str:
            # QSetting stores boolean as string in ini file
            return False if value.lower() == 'false' else True

        return value

    def setValue(self, key, value, default=False):
        """Set value of a setting key

        :param key: setting key
        :type key: Enum
        :param value: new value
        :type value: Any
        :param default: flag indicating default should also be set
        :type default: bool
        """
        self.local[key.value] = value
        if default:
            self.system.setValue(key.value, value)

    def reset(self, default=False):
        """ Clear saved values of setting keys that belong to a Group. Keys without
        a group e.g. Check_Update are not cleared.

        :param default: flag indicating default should also be reset
        :type default: bool
        """
        self.local.clear()
        if default:
            for group in self.Group:
                self.system.remove(group.value)

    def filename(self):
        """ Returns full path of setting file

        :return: setting file path
        :rtype: str
        """
        return self.system.fileName()


settings = Setting()
LOG_PATH = pathlib.Path(settings.filename()).parent / 'logs'


def setup_logging(filename):
    """
    Configure of logging file handler.

    :param filename: name of log file
    :type filename: str
    """
    try:
        LOG_PATH.mkdir(parents=True, exist_ok=True)

        with open(LOG_CONFIG_PATH, 'rt') as config_file:
            config = json.load(config_file)
            config['handlers']['file_handler']['filename'] = LOG_PATH / filename
            logging.config.dictConfig(config)
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Could not initialize logging with %s", LOG_CONFIG_PATH)
