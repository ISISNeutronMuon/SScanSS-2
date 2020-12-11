##########
File Types
##########
The various file types used by SScanSS-2 are described below.

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
The example file below will setup strain component measurement vectors aligned in the +X and –Z directions for a two
detector instrument at 4 measurement points.::

    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000
    1.0000000 0.0000000 0.0000000 -0.0000000 -0.0000000 -1.0000000

.. note::
    Append more vectors to the end of the file to add secondary vector alignments. When the number of measurement
    vectors are greater than the number of measurement points, the extra vectors will be considered as secondary
    vector alignments.

.. _trans file:

******************************
Transformation matrix (.trans)
******************************
The *.trans* file is a simple comma or space delimited text file containing a 4 X 4 matrix. The transformation matrix should
contain a translation and rotation only to be considered valid. An example file is shown below::

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

Where "Fi" is the index of fiducial point, "xi yi  zi" are the coordinates of that fiducial point in the instrument
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
