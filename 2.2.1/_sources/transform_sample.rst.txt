################
Transform Sample
################
It may be convenient to change the coordinate system of a sample model, for example, if it will simplify the
specification of measurement point positions. Transformations will be applied to sample and to all attached objects
(which includes fiducial points, measurement points and vectors if present). The software provides several options for
transforming samples.

.. warning::
   Do not forget to add fiducial points before applying a transformation to the sample otherwise the
   sample could be misaligned on the instrument when fiducials are used.

.. note::
    Transformations are applied directly to sample (as opposed to being stored) so when no sample exist all the
    transform windows will be disabled.

*************
Rotate sample
*************
Click the rotate sample |rotate| button on the toolbar. The rotate sample window will be opened. Type in the desired
X, Y, Z axis rotation and click the **Rotate** button.

.. image:: images/rotate_sample.png
   :scale: 80
   :alt: Rotate Sample Window
   :align: center

****************
Translate sample
****************
Click the translate sample |translate| button on the toolbar. The translate sample window will be opened. Type in the
desired X, Y, Z axis translation and click the **Translate** button.

.. image:: images/translate_sample.png
   :scale: 80
   :alt: Translate Sample Window
   :align: center

****************************
Transform sample with matrix
****************************
Click the transform sample |transform| button on the toolbar. The transform sample window will be opened.

.. image:: images/transform_sample.png
   :scale: 80
   :alt: Transform Sample Window
   :align: center

* Click the **Load Matrix** button and select the :ref:`trans file` that contains the transformation matrix.
* A preview of the file will be shown in window, to use the inverse of the loaded matrix click
  the **Invert Transformation Matrix** check box
* Click the **Apply Transform** button

*********************
Move origin to sample
*********************
It may be useful to move the scene origin to center or edge of the sample. To do this click the move origin to sample
|origin| button on the toolbar.

.. image:: images/move_origin.png
   :scale: 80
   :alt: Move Origin to Sample Window
   :align: center

* Select the boundary to move the origin to.
* Select the Axis to Ignore i.e. leave fixed.
* Click the **Move Origin** button

.. tip::
    If the sample contains extraneous surfaces for example, parts of a table may be scanned with the sample, using
    this may not place the origin at the expected position since the extraneous surface may now constitute part of the
    boundary. In such a case, remove extraneous surfaces in a mesh editing software beforehand.

********************************
Rotate sample by plane alignment
********************************
It may be useful to rotate the sample so that a plane on the sample is aligned with a plane in the world. To do this
click the rotate sample by plane alignment |plane| button on the toolbar.

.. image:: images/plane_alignment.png
   :scale: 80
   :alt: Rotate Sample by Plane Alignment Window
   :align: center

* Activate point selection by clicking the |point| button.
* Select a minimum of 3 points to estimate a plane.
* Specify a final plane in the world coordinate frame.
* Click the **Align Planes** button.

Rotating and panning the scene is disabled (zooming is still enabled) when point selection is active, click the
|select| button to activate scene navigation. To delete a point, select the point from the point list (the
corresponding graphics will be highlighted) and click the |delete| button.

.. tip::
    You can select points from the point list while point selection is active. Select multiple points
    using **Shift + Left Click** or **Ctrl+ Left Click**

.. |rotate| image:: images/rotate.png
            :scale: 10

.. |translate| image:: images/translate.png
            :scale: 10

.. |transform| image:: images/transform-matrix.png
            :scale: 10

.. |origin| image:: images/origin.png
            :scale: 10

.. |plane| image:: images/plane_align.png
            :scale: 10

.. |select| image:: images/select.png
            :scale: 10

.. |point| image:: images/point.png
            :scale: 10

.. |delete| image:: images/cross.png
            :scale: 10