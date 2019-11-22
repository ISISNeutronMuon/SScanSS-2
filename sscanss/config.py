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
STATIC_PATH = SOURCE_PATH / 'static'
IMAGES_PATH = STATIC_PATH / 'images'
LOG_CONFIG_PATH = SOURCE_PATH / 'logging.json'


def path_for(filename):
    return str((IMAGES_PATH / filename).as_posix())


@unique
class Group(Enum):
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


__defaults__ = {Key.Geometry: bytearray(b''), Key.Check_Update: True, Key.Recent_Projects: [],
                Key.Local_Max_Eval: 1000, Key.Global_Max_Eval: 200, Key.Align_First: True,
                Key.Angular_Stop_Val: 1.00, Key.Position_Stop_Val: 1e-2,
                Key.Sample_Colour: (0.65, 0.65, 0.65, 1.0),
                Key.Fiducial_Colour: (0.4, 0.9, 0.4, 1.0), Key.Fiducial_Disabled_Colour: (0.9, 0.4, 0.4, 1.0),
                Key.Measurement_Colour: (0.01, 0.44, 0.12, 1.0), Key.Measurement_Disabled_Colour: (0.9, 0.4, 0.4, 1.0),
                Key.Vector_1_Colour: (0.0, 0.0, 1.0, 1.0), Key.Vector_2_Colour: (1.0, 0.0, 0.0, 1.0),
                Key.Selected_Colour: (0.94, 0.82, 0.68, 1.0), Key.Cross_Sectional_Plane_Colour: (0.93, 0.83, 0.53, 1.0),
                Key.Fiducial_Size: 5, Key.Measurement_Size: 5, Key.Vector_Size: 10}


class Setting:
    def __init__(self):
        self._setting = QtCore.QSettings(QtCore.QSettings.IniFormat,
                                         QtCore.QSettings.UserScope,
                                         'SScanSS 2', 'SScanSS 2')
        self.Key = Key
        self.Group = Group

    def value(self, key):
        default = __defaults__.get(key, None)
        if default is None:
            return self._setting.value(key.value)

        value = self._setting.value(key.value, default)
        if type(default) is int:
            return int(value)
        if type(default) is float:
            return float(value)
        if type(default) is bool and type(value) is str:
            return False if value.lower() == 'false' else True

        return value

    def setValue(self, key, value):
        self._setting.setValue(key.value, value)

    def reset(self):
        for group in self.Group:
            self._setting.remove(group.value)

    def filename(self):
        return self._setting.fileName()


settings = Setting()
LOG_PATH = pathlib.Path(settings.filename()).parent / 'logs'


def setup_logging(filename):
    """
    Configure of logging file handler.

    :param filename: name of log file
    :type filename: str
    """
    try:
        if not LOG_PATH.exists():
            LOG_PATH.mkdir(parents=True)

        with open(LOG_CONFIG_PATH, 'rt') as config_file:
            config = json.load(config_file)
            config['handlers']['file_handler']['filename'] = LOG_PATH / filename
            logging.config.dictConfig(config)
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Could not initialize logging with %s", LOG_CONFIG_PATH)
