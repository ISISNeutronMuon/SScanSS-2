##########
File Types
##########
The various file types used by SScanSS 2 are described below.

.. _fiducial file:

*******************************
Fiducial point file (.fiducial)
*******************************
The *.fiducial* file is a comma or space delimited text file. The file must contain three columns representing the
X, Y, Z coordinates of the fiducial point and an optional fourth row which specifies enabled status. All point are
enabled by default if not explicitly disabled.

A comma delimited example with all points enabled::

    0.0000000,	0.0000000,	0.0000000
    0.0000000,	10.0000000,	0.0000000
    0.0000000,	-10.0000000,    0.0000000

A space delimited example with 1\ :sup:`st` \ and 3\ :sup:`rd` \ point enabled::

    0.0000000	0.0000000	0.0000000
    0.0000000	10.0000000	0.0000000	False
    0.0000000	-10.0000000	0.0000000	True

.. _measurement file:

*******************************
Measurement file (.measurement)
*******************************
The *.measurement* file is a comma or space delimited text file with the exact same structure as the `Fiducial point file (.fiducial)`_.

.. _vector file:

*******************************
Measurement vector file (.vecs)
*******************************
The *.vecs* file is a is a simple comma or space delimited text file of the form:

::

    x11 y11 z11 . . . x1N y1N z1N
    x21 y21 z21 . . . x2N y2N z2N
    :
    xM1 yM1 zM1 . . . xMN yMN zMN

Where "M" is the number of measurement points and "N" is the number of detectors.
The example file below will setup strain component or measurement vectors aligned in the +X and –Z directions for a two
detector instrument at 4 measurement points.::

    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000

Append more vectors to the end of the file to add secondary vector alignments. When the number of measurement vectors
are greater than the number of measurement points, the extra vectors will be considered as secondary vector alignments.
Measurement vectors must be zero vectors or have a magnitude of 1 accurate to 7 decimal digits to be valid.

.. _angle file:

**************************
Euler angle file (.angles)
**************************
The *.angles* file is a is a simple comma or space delimited text file of the form:

::

    xyz
    x1 y1 z1
    x2 y2 z2
    :
    xM yM zM

Where "M" is the number of angles, the header is a string that indicates the order of the angles ("xyz" and "zyx" are
supported), and "xi yi zi" are the euler angles in degrees in the order specified in the header.
With the example file below, measurement vectors can be created using Euler angles.::

    zyx
    -30.0 35.0 0.0
    -30.0 15.0 0.0
    -30.0 0.0 0.0
    -30.0 -15.0 0.0

Append more angles to the end of the file to add secondary vector alignments. When the number of angles
are greater than the number of measurement points, the extra vectors created will be considered as secondary vector
alignments.

.. _trans file:

***********************************
Transformation matrix file (.trans)
***********************************
The *.trans* file is a simple comma or space delimited text file containing a 4 X 4 matrix. The transformation matrix
should contain a translation and rotation only to be considered valid (The rotation vectors should have a magnitude of
1 accurate to 7 decimal digits). An example file is shown below::

    1.0000000   0.0000000   0.0000000   0.0000000
    0.0000000   1.0000000   0.0000000   0.0000000
    0.0000000   0.0000000   1.0000000   0.0000000
    0.0000000   0.0000000   0.0000000   1.0000000

.. _fpos file:

*************************************
Fiducial point positions file (.fpos)
*************************************
The *.fpos* file is a simple comma or space delimited text file of the form::

    Fi  xi  yi  zi  P1i P2i P3i P4i . . . PNi
    Fj  xj  yj  zj  P1j P2j P3j P4j . . . PNj
    :
    Fk  xk  yk  zk  P1k P2k P3k P4k . . . PNk

Where "Fi" is the index of fiducial point, "xi yi zi" are the coordinates of that fiducial point in the instrument
coordinate system and "P1i P2i P3i Pi4i" are the positioning system variables at the time the fiducial point position
were measured.

The example below is for the measurement of 5 fiducial points in the order (3,1,5,2,4) on an x, y, z, omega positioning
system, with a rotation in omega of 90 degrees between the measurements of the 5\ :sup:`th` \ and 2\ :sup:`nd` \
fiducial point.::

    3   -72.2471016 -9.3954792 -0.1111806 270.0 -200.0 600.0 90.0
    1   20.8212362  -9.3734531 70.5337096 270.0 -200.0 600.0 90.0
    5   26.9184367  -9.1761998 -68.1036148 270.0 -200.0 600.0 90.0
    2   -56.8372955 -9.5676188 46.7159424 270.0 -200.0 600.0 0.0
    4   -49.1049504 -9.3734125 -54.1452751 270.0 -200.0 600.0 0.0


.. _calib file:

*************************************
Robot world calibration file (.calib)
*************************************
The *.calib* file is a simple comma or space delimited text file of the form::

    Pi  Fi  xi  yi  zi  V1i V2i V3i V4i . . . VNi
    Pj  Fj  xj  yj  zj  V1j V2j V3j V4j . . . VNj
    :
    Pk  Fk  xk  yk  zk  V1k V2k V3k V4k . . . VNk

Where "Pi"is the index of the pose (which should begin at 1),  "Fi" is the index of fiducial point, "xi yi  zi" are the
coordinates of that fiducial point in the instrument coordinate system and "P1i P2i P3i Pi4i" are the positioning
system variables at the time the fiducial point position were measured.

The example below is for the calibration of a 3 degree of freedom revolute robot, several fiducials point are measured
in 3 different poses.::

    1	1	102.8377418	 -81.96943728	-363.7358	    90	-90	  50
    1	2	79.42193655	 8.908326571	-417.8232923	90	-90	  50
    1	3	48.9408	     131.9627569	-492.119	    90	-90	  50
    1	4	12.97246409	 62.3434793	    -423.5625153	90	-90	  50
    2	1	42.9476	     74.26894944	-329.0102	   -90	 90	 -50
    2	2	18.8411894	 -16.465928	    -383.035	   -90	 90	 -50
    2	3	-15.3111639	 -139.5736584	-455.65207	   -90	 90	 -50
    2	4	53.03868877	 -61.486	    -447.1121114   -90	 90	 -50
    2	5	25.472906	 -167.38232	    -510.7875906   -90	 90	 -50
    2	6	58.79880665	 -136.3461627	-508.2237055   -90	 90	 -50
    3	1	74.46685385	 -130.4356927	-485.11675	    90	 180  90
    3	2	75.08461883	 -22.10217867	-485.0733318	90	 180  90
    3	3	75.7352316	 124.8278	    -486.6431378	90	 180  90
    3	4	69.81	     39.9879	    -426.605	    90	 180  90
    3	5	70.33116112	 166.5633859	-425.7904619	90	 180  90
    3	6	70.17106437	 131.8591781	-396.1656491	90	 180  90

