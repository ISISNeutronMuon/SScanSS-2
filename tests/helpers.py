from PyQt5.QtWidgets import QMainWindow, QApplication

APP = QApplication([])


def do_nothing(*_args, **_kwargs):
    pass


class TestSignal:
    def __init__(self):
        self.call = do_nothing

    def connect(self, call):
        self.call = call

    def emit(self, *args):
        self.call(*args)


class TestView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.presenter = None
        self.scenes = None
        self.showSelectChoiceMessage = None
        self.showMessage = do_nothing
        self.showPathLength = do_nothing
        self.showScriptExport = do_nothing


SAMPLE_IDF = '''{
    "instrument":{
        "name": "GENERIC",
        "version": "1.0",
        "gauge_volume": [0.0, 0.0, 0.0],
        "incident_jaws":{
            "beam_direction": [1.0, 0.0, 0.0],
            "beam_source": [-300.0, 0.0, 0.0],
            "aperture": [1.0, 1.0],
            "aperture_lower_limit": [0.5, 0.5],
            "aperture_upper_limit": [15.0, 15.0],
            "positioner": "incident_jaws",
            "visual":{
                    "pose": [300.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
            }
        },
        "detectors":[
            {
                "name":"Detector",
                "default_collimator": "Snout 25mm",
                "positioner": "diffracted_jaws",
                "diffracted_beam": [0.0, 1.0, 0.0]
            }
        ],
        "collimators":[
            {
                "name": "Snout 25mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 50mm",
                "detector": "Detector",
                "aperture": [2.0, 2.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 100mm",
                "detector": "Detector",
                "aperture": [1.0, 1.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            },
            {
                "name": "Snout 150mm",
                "detector": "Detector",
                "aperture": [4.0, 4.0],
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
                    "mesh": "model_path",
                    "colour": [0.47, 0.47, 0.47]
                }
            }
        ],
        "positioning_stacks":[
        {
            "name": "Positioning Table Only",
            "positioners": ["Positioning Table"]
        },
        {
            "name": "Positioning Table + Huber Circle",
            "positioners": ["Positioning Table", "Huber Circle"]
        }
        ],
        "positioners":[
            {
                "name": "Positioning Table",
                "base": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "custom_order": ["X Stage", "Y Stage", "Omega Stage"],
                "joints":[
                    {
                        "name": "X Stage",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -201.0,
                        "upper_limit": 192.0,
                        "parent": "y_stage",
                        "child": "x_stage"
                    },
                    {
                        "name": "Y Stage",
                        "type": "prismatic",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -101.0,
                        "upper_limit": 93.0,
                        "parent": "omega_stage",
                        "child": "y_stage"
                    },
                    {
                        "name": "Omega Stage",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -170.0,
                        "upper_limit": 166.0,
                        "parent": "base",
                        "child": "omega_stage"
                    }],
                "links": [
                    {"name": "base"},
                    {"name": "omega_stage"},
                    {"name": "y_stage"},
                    {"name": "x_stage"}
                ]
            },
            {
                "name": "Huber Circle",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Chi",
                        "type": "revolute",
                        "axis": [0.0, 1.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 300.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "chi_axis"
                    },
                    {
                        "name": "Phi",
                        "type": "revolute",
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -360.0,
                        "upper_limit": 360.0,
                        "parent": "chi_axis",
                        "child": "phi_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {"name": "chi_axis"},
                    {"name": "phi_axis"}
                ]
            },
            {
                "name": "incident_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Jaws X Axis",
                        "type": "prismatic",
                        "axis": [1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -800.0,
                        "upper_limit": 0.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "jaw_x_axis"
                    }

                ],
                "links": [
                    {"name": "base"},
                    {
                        "name": "jaw_x_axis", 
                        "visual":
                        {
                            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "mesh": "model_path",
                            "colour": [0.78, 0.39, 0.39]
                        }
                    }
                ]
            },
            {
                "name": "diffracted_jaws",
                "base":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joints": [
                    {
                        "name": "Angular Axis",
                        "type": "revolute",
                        "axis": [0.0, 0.0, -1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": -120.0,
                        "upper_limit": 120.0,
                        "home_offset": 0.0,
                        "parent": "base",
                        "child": "angular_axis"
                    },
                    {
                        "name": "Radial Axis",
                        "type": "prismatic",
                        "axis": [-1.0, 0.0, 0.0],
                        "origin": [0.0, 0.0, 0.0],
                        "lower_limit": 0.0,
                        "upper_limit": 100.0,
                        "home_offset": 0.0,
                        "parent": "angular_axis",
                        "child": "radial_axis"
                    }
                ],
                "links": [
                    {
                        "name": "base", 
                        "visual":
                        {
                            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "mesh": "model_path",
                            "colour": [0.78, 0.39, 0.39]
                        }
                    },
                    {"name": "angular_axis"},
                    {"name": "radial_axis"}
                ]
            }			
        ],
        "fixed_hardware":[
            {
                "name":  "monochromator",
                "visual":{
                    "pose": [0.0, 0.0, 0.0, 90.0, 90.0, 0.0],
                    "mesh": "model_path",
                    "colour": [0.16, 0.39, 0.39]
                }
            },
            {
                "name": "floor",
                "visual":{
                    "pose": [0.0, 0.0, -15.0, 0.0, 0.0, 0.0],
                    "mesh": "model_path",
                    "colour": [0.7, 0.7, 0.7]
                }
            }
        ]
    }
}'''
