from enum import Enum, unique
from PyQt5 import QtCore

__version__ = '0.0.1'


@unique
class Group(Enum):
    Graphics = 'Graphics'
    Simulation = 'Simulation'


@unique
class Key(Enum):
    Geometry = 'Geometry'
    Window_State = 'Window_State'
    Recent_Projects = 'Recent_Projects'
    Stop_Val = f'{Group.Simulation.value}/Stop_Val'
    Local_Max_Eval = f'{Group.Simulation.value}/Local_Max_Eval'
    Global_Max_Eval = f'{Group.Simulation.value}/Global_Max_Eval'
    Sample_Colour = f'{Group.Graphics.value}/Sample_Enabled_Colour'
    Fiducial_Colour = f'{Group.Graphics.value}/Fiducial_Colour'


__defaults__ = {Key.Geometry: bytearray(b''), Key.Window_State: bytearray(b''), Key.Recent_Projects: [],
                Key.Local_Max_Eval: 1000, Key.Global_Max_Eval: 100, Key.Stop_Val: 1e-2,
                Key.Sample_Colour: (0.65, 0.65, 0.65, 1.0), Key.Fiducial_Colour: (0.4, 0.9, 0.4, 1.0)}


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

        return value

    def setValue(self, key, value):
        self._setting.setValue(key.value, value)

    def reset(self):
        for group in self.Group:
            self._setting.remove(group.value)


settings = Setting()
