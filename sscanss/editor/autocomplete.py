import enum
from collections import namedtuple

AutocompletionObject = namedtuple('AutocompletionObject',['Key', 'Type', 'Optional', 'Description'])

class GenericInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name','string','Required','Unique name of instrument')
    VERSION = AutocompletionObject('version','string','Required','Version number of file')
    SCRIPT_TEMPLATE = AutocompletionObject('script_template','string','Optional (generic)', 'Path of script template')
    GAUGE_VOLUME = AutocompletionObject('gauge_volume','array of float', 'Required', 'Position of gauge volume')
    INCIDENT_JAWS = AutocompletionObject('incident_jaws','Jaws Object', 'Required', 'Jaws of instrument')
    DETECTORS = AutocompletionObject('detectors', 'array of Detector Objects', 'Required', 'Detectors of instrument')
    COLLIMATORS = AutocompletionObject('collimators', 'array of Collimator Objects', 'Optional (None)', 'Collimators of instrument')
    POSITIONING_STACKS = AutocompletionObject('positioning_stacks', 'array of Positioning Stack Objects', 'Required', 'Positioning stacks of instrument')
    POSITIONERS = AutocompletionObject('positioners', 'array of Positioner Objects', 'Required', 'Positioners of instrument')
    FIXED_HARDWARE = AutocompletionObject('fixed_hardware', 'array of Fixed Hardware Objects', 'Optional (None)', 'Fixed hardware on instrument')

class PositionerInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name','string','Required','Unique name of positioner')
    BASE = AutocompletionObject('base', 'array of floats', 'Optional (zero array)', 'Base matrix of the positioner as a 6D array. First three value should be XYZ translation and next three should be XYZ orientation in Degrees')
    TOOL = AutocompletionObject('tool', 'array of floats', 'Optional (zero array)', 'Tool matrix of the positioner as a 6D array. First three value should be XYZ translation and next three should be XYZ orientation in Degrees')
    CUSTOM_ORDER = AutocompletionObject('custom_order', 'array of strings', 'Optional (None)', 'Order of joint if order is different from kinematics')
    JOINTS = AutocompletionObject('joints', 'array of Joint Objects', 'Required', 'Joints of the positioner')
    LINKS = AutocompletionObject('links', 'array of Link Objects', 'Required', 'Links of the positioner')

class JointInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object. The joints in a positioner must have unique names')
    TYPE = AutocompletionObject('type', 'enum [prismatic, revolute]', 'Required', 'The joint type: revolute for rotating joints and prismatic for translating joints')
    PARENT = AutocompletionObject('parent', 'string', 'Required', 'The name of the link object to which the joint is attached')
    CHILD = AutocompletionObject('child', 'string', 'Required', 'The name of the link object that is attached to the joint')
    AXIS = AutocompletionObject('axis', 'array of floats', 'Required', 'The axis of translation or rotation with respect to the instrument coordinate frame')
    ORIGIN = AutocompletionObject('origin', 'array of floats', 'Required', 'The centre of rotation for the revolute joint or the start position of prismatic joints with respect to the instrument coordinate frame')
    LOWER_LIMIT = AutocompletionObject('lower_limit', 'float', 'Required', 'The lower limit of the joint')
    UPPER_LIMIT = AutocompletionObject('upper_limit', 'float', 'Required', 'The upper limit of the joint')
    HOME_OFFSET = AutocompletionObject('home_offset', 'float', 'Optional ((upper_limit+lower_limit)/2)', 'The initial offset value of the manipulator')

class LinkInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object. The links in a positioner must have unique names')
    VISUAL = AutocompletionObject('visual', 'Visual Object', 'Optional (None)', 'Visual representation of lobject')

class VisualInstrumentAutocomplete(enum.Enum):
    POSE = AutocompletionObject('pose', 'array of floats', 'Optional (zero array)', 'Transform to apply to the mesh as a 6D array. First three value should be XYZ translation and next three should be XYZ orientation in Degrees')
    COLOUR = AutocompletionObject('colour', 'array of floats', 'Optional (zero array)', 'Normalized RGB colour [0-1]')
    MESH = AutocompletionObject('mesh', 'string', 'required', 'Relative file path to mesh')

class PositioningStackInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object')
    POSITIONERS = AutocompletionObject('positioners', 'array of strings', 'Required', 'Names of positioners in the stack from bottom to top')

class DetectorInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object')
    DEFAULT_COLLIMATOR = AutocompletionObject('default_collimator', 'string', 'Optional (None)', 'Name of the default collimator')
    DIFFRACTED_BEAM = AutocompletionObject('diffracted_beam', 'array of floats', 'Required', 'Normalized vector of the diffracted beam')
    POSITIONER = AutocompletionObject('positioner', 'string', 'Optional (None)', 'Name of the positioner the detector is attached to')

class CollimatorInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object')
    DETECTOR = AutocompletionObject('detector', 'string', 'Required', 'Name of detector the collimator is attached to')
    APERTURE = AutocompletionObject('aperture', 'array of floats', 'Required', 'Horizontal and vertical size of collimator aperture')
    VISUAL = AutocompletionObject('visual', 'Visual Object', 'Required', 'Visual representation of object')

class JawsInstrumentAutocomplete(enum.Enum):
    APERTURE = AutocompletionObject('aperture', 'array of floats', 'Required', 'Horizontal and vertical size of jaws aperture')
    APERTURE_LOWER_LIMIT = AutocompletionObject('aperture_lower_limit', 'array of floats', 'Required', 'Horizontal and vertical lower limit of jaws')
    APERTURE_UPPER_LIMIT = AutocompletionObject('aperture_upper_limit', 'array of floats', 'Required', 'Horizontal and vertical upper limit of jaws')
    BEAM_DIRECTION = AutocompletionObject('beam_direction', 'array of floats', 'Required', 'Normalized vector indicating the direction of beam from source')
    BEAM_SOURCE = AutocompletionObject('beam_source', 'array of floats', 'Required', 'Source position of the beam')
    POSITIONER = AutocompletionObject('positioner', 'string', 'Optional (None)', 'Name of positioner the jaws are attached to')
    VISUAL = AutocompletionObject('visual', 'Visual Object', 'Required', 'Visual representation of object')

class FixedHardwareInstrumentAutocomplete(enum.Enum):
    NAME = AutocompletionObject('name', 'string', 'Required', 'Unique name of object')
    VISUAL = AutocompletionObject('visual', 'Visual Object', 'Required', 'Visual representation of object')

instrument_autocompletions = [GenericInstrumentAutocomplete, PositionerInstrumentAutocomplete, JointInstrumentAutocomplete, LinkInstrumentAutocomplete, VisualInstrumentAutocomplete, PositioningStackInstrumentAutocomplete, DetectorInstrumentAutocomplete, CollimatorInstrumentAutocomplete, JawsInstrumentAutocomplete, FixedHardwareInstrumentAutocomplete]