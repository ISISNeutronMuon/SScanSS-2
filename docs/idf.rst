#################################
Instrument Description File (IDF)
#################################
An instrument definition file (IDF) is a |json| file that describes an instrument, and provides details about those components
of the instruments that are essential for positioning the sample such the positioning system, guage volume, detector etc.
SScanSS is mainly concerned with the geometry i.e. shape and kinematic of these components.

.. note::
    The instrument description file used in SScanSS is different from the one used by the |mantid| project

.. warning::
    The keywords in the instrument description file are CASE-SENSITIVE so use the same case as in the documentation

================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of instrument
version            string                             Required                 Version number of file
script_template    string                             Optional (generic)       Path of script template
gauge_volume       array of float                     Required                 Position of gauge volume
incident_jaws      :ref:`jaws object`                 Required                 Jaws of instrument
detectors          array of :ref:`Detector            Required                 Detectors of instrument
                   Objects <detector object>`
collimators        array of :ref:`Collimator          Optional (None)          Collimators of instrument
                   Objects <collimator object>`
positioning_stacks array of :ref:`Positioning Stack   Required                 Positioning stacks of instrument
                   Objects <positioning stack
                   object>`
positioners        array of :ref:`Positioner          Required                 Positioners of instrument
                   Objects <positioner object>`
fixed_hardware     array of :ref:`Fixed Hardware      Optional (None)          Fixed hardware on instrument
                   Objects <fixed hardware object>`
================== ================================== ======================== ===========

*****************
Positioner Object
*****************

================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of positioner
base               array of floats                    Optional (zero array)    Base matrix of the positioner as a 6D array.
                                                                               First three value should be XYZ translation
                                                                               and next three should be XYZ orientation in Degrees
tool               array of floats                    Optional (zero array)    Tool matrix of the positioner as a 6D array.
                                                                               First three value should be XYZ translation
                                                                               and next three should be XYZ orientation in Degrees
custom_order       array of strings                   Optional (None)          Order of joint if order is different from kinematics.
joints             array of :ref:`Joint               Required                 Joints of the positioner
                   Objects <joint object>`
links              array of :ref:`Link                Required                 Links of the positioner
                   Objects <link object>`
================== ================================== ======================== ===========

************
Joint Object
************

================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object.
                                                                               The joints in a positioner must have unique names
description        string                             Optional                 Description of the joint object
type               enum [prismatic, revolute]         Required                 The joint type: revolute for rotating joints
                                                                               and prismatic for translating joints.
parent             string                             Required                 The name of the link object to which the joint is attached
child              string                             Required                 The name of the link object that is attached to the joint
axis               array of floats                    Required                 The axis of translation or rotation with
                                                                               respect to the instrument coordinate frame.
origin             array of floats                    Required                 The centre of rotation for the revolute joint
                                                                               or the start position of prismatic joints with
                                                                               respect to the instrument coordinate frame.
lower_limit        float                              Required                 The lower limit of the joint
upper_limit        float                              Required                 The upper limit of the joint
home_offset        float                              Optional ((upper_limit   The initial offset value of the manipulator.
                                                      + lower_limit)/2)
================== ================================== ======================== ===========

***********
Link Object
***********
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object.
                                                                               The links in a positioner must have unique names
visual             :ref:`visual object`               Optional (None)          Visual representation of lobject
================== ================================== ======================== ===========

*************
Visual Object
*************
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
pose                array of floats                   Optional (zero array)    Transform to apply to the mesh as a 6D array.
                                                                               First three value should be XYZ translation and
                                                                               next three should be XYZ orientation in Degrees
colour              array of floats                   Optional (zero array)    Normalized RGB colour [0 - 1]
mesh                string                            Required                 Relative file path to mesh
================== ================================== ======================== ===========

************************
Positioning Stack Object
************************
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object.
positioners        array of strings                   Required                 Names of positioners in the stack from bottom to top.
================== ================================== ======================== ===========

***************
Detector Object
***************
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object
default_collimator string                             Optional (None)          Name of the default collimator
diffracted_beam    array of floats                    Required                 Normalized vector of the diffracted beam
positioner         string                             Optional (None)          Name of positioner the detector is attached to.
================== ================================== ======================== ===========

*****************
Collimator Object
*****************
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object
detector           string                             Required                 Name of detector the collimator is attached to
aperture           array of floats                    Required                 Horizontal and vertical size of collimator’s aperture
visual             :ref:`visual object`               Required                 Visual representation of object
================== ================================== ======================== ===========


***********
Jaws Object
***********
==================== ================================== ======================== ===========
Key                  Type                               Optional (Default Value) Description
==================== ================================== ======================== ===========
aperture             array of floats                    Required                 Horizontal and vertical size of jaws’ aperture
aperture_lower_limit array of floats                    Required                 Horizontal and vertical lower limit of jaws
aperture_upper_limit array of floats                    Required                 Horizontal and vertical upper limit of jaws
beam_direction       array of floats                    Required                 Normalized vector indicating the direction of beam from source
beam_source          array of floats                    Required                 Source position of the beam
positioner           string                             Optional (None)          Name of positioner the jaws are attached to.
visual               :ref:`visual object`               Required                 Visual representation of object
==================== ================================== ======================== ===========

*********************
Fixed hardware Object
*********************
================== ================================== ======================== ===========
Key                Type                               Optional (Default Value) Description
================== ================================== ======================== ===========
name               string                             Required                 Unique name of object
visual             :ref:`visual object`               Required                 Visual representation of object
================== ================================== ======================== ===========

.. |json| raw:: html

   <a href="http://www.json.org/" target="_blank">JSON</a>


.. |mantid| raw:: html

   <a href="https://www.mantidproject.org/" target="_blank">Mantid</a>
