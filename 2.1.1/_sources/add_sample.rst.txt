#############
Insert Sample
#############
Sample models may be imported from a 3D model file, or creating from primitives. Both options are detailed below.

***************************
Import a 3D model from file
***************************
The 3D model for a Sample may be imported from ASCII/binary STL files or Wavefront OBJ files. Click
**Insert > Sample > File** and browse to the location of the 3D model and select it. The mesh will be cleaned
(repeated vertices and faces with zero area will be removed) and the model will displayed in the graphics window.

.. tip::
   STL files in ASCII format will take more time to load than binary files but
   performance after loading will not be affected.

******************************
Create a model from primitives
******************************
Simple models may be generated from primitives, e.g. cuboids, spheres, tubes, cone etc. Click
**Insert > Sample > primitives** and select the desired primitive. Type in the required parameters for defining
the primitive and click the **Create** button.

.. image:: images/insert_primitive.png
   :scale: 80
   :alt: Insert Primitive Window
   :align: center

.. tip::
    Use the |arrow| button to quickly switch to another primitive.

*****************
Add another model
*****************
In some cases, a second model may be needed to properly approximate the sample or a holder. After adding the first sample,
to add a second model simply follow the steps above for file or primitive, you will be asked if the new model should replace
the current one or be combined with it.

.. image:: images/another_model.png
   :scale: 80
   :alt: Add model confirmation dialog
   :align: center

.. note::
    Combining models is only possible for mesh samples not volumes.

*********************
View model properties
*********************
In order to view the properties of the sample click **View > Other Windows > Sample Properties**.

For mesh samples- the memory (in mb), faces and vertices are displayed. 

.. image:: images/mesh_sample_properties.png
   :scale: 80
   :alt: Mesh sample properties dialog 
   :align: center

For volume samples- the memory (in mb), dimensions, voxel size, minimum intensity and maximum intensity are displayed. 

.. image:: images/volume_sample_properties.png
   :scale: 80
   :alt: Volume sample properties dialog 
   :align: center

.. note::
    Use the **Ctrl+Shift+I** keyboard shortcut for quick access to the sample properties information.

***************
Export a sample
***************
The 3D model for a Sample can be exported from project file to a STL file. Click **File > Export... > Sample** in the
main menu, navigate to the desired save location in the file dialog, enter a name for the file and press the **Save**
button.


.. |arrow| image:: images/arrow.png
            :scale: 10

.. |delete| image:: images/cross.png
            :scale: 10

.. |merge| image:: images/merge.png
            :scale: 10

.. |change main| image:: images/merge.png
            :scale: 10
