##########################
Align Sample on Instrument
##########################
Sample preparation is typically performed off the instrument after which the sample is moved to the instrument. SScanSS needs
to know the transformation (alignment matrix) from the original sample pose to the pose on the instrument. Once the alignment
matrix is specified, the sample model, fiducials, and measurements will be rendered in the instrument scene. The alignment
matrix can be specified in the following ways:

.. tip::
    An alignment matrix must be specified even if it is an identity matrix without which the simulation cannot
    be executed.

******************
Align with 6D pose
******************
The alignment matrix can be specified with 6 values: 3 (X, Y, Z) rotation angles and 3 (X, Y, Z) translation values. To
do this click **Instrument > 'Align Sample on Instrument > 6D Pose**, type in the values, and click **Align Sample**.

.. image:: images/align_with_6D_pose.png
   :scale: 80
   :alt: Align with 6D pose Window
   :align: center

********************************
Align with transformation matrix
********************************
The alignment matrix can be specified directly by importing a :ref:`trans file`. To do this click
**Instrument > 'Align Sample on Instrument > Transformation Matrix**, select the file in the dialog and click open.

**************************
Align with fiducial points
**************************
A common method of determining the alignment matrix is to re-measure fiducial points (a minimum of 3 points is required)
on the instrument and calculate the rigid transformation directly.

* Write re-measured fiducial points into a :ref:`fpos file`.
* Click **Instrument > 'Align Sample on Instrument > Fiducial Points**.
* Select the :ref:`fpos file` and click open.

On the Error report

* If accuracy is poor, disable the point with the largest errror and click **Recalculate**.
* Repeat previous step until accuracy is tolerable or only 3 points are enabled.
* Click **Accept** if the accuracy is within tolerance.

.. image:: images/alignment_error.png
   :scale: 80
   :alt: Alignment error dialog
   :align: center

.. tip::
    Use more than 3 points to get a better approximation of the alignment matrix.
