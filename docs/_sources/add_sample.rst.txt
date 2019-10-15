#############
Insert Sample
#############

Sample models may be imported from a 3D model file, or creating from primitives. Both options are detailed below

*************************
Loading an 3D model file
*************************
The 3D model for a Sample may be imported from ASCII/binary STL files or Wavefront OBJ files. Click
**Insert > Sample > File** and browse to the location of the 3D model and select it. The mesh will be cleaned
(repeated vertices and faces with zero area will be removed) and the model will displayed in the graphics window.

.. tip::
   STL files in ASCII format will take more time to load than binary files but
   performance after loading will not be affected.

******************************
Create a model from primitives
******************************
Simple models may be generated from primitives, e.g. cuboids, spheres, tubes, etc. Click
**Insert > Sample > primitives** and select the desired primitive.

*****************
Add another model
*****************

****************
Managing samples
****************
Basic sample management can be performed via the sample manager. The sample manager will be opened when a new sample is
added, if the sample manager is closed it can be opened by selecting **View > Other Windows > Samples** in the menu.
Samples can be merged, deleted or designated as main sample from the manager, and these operation can be undone (Ctrl+Z)
if needed. The sample manager displays a list of sample names, primitive generated samples can be named on creation while
imported samples will be given a name generated from its filename. Selecting a sample from the manager will cause
the 3D model to be highlighted in the graphics window, this can be used to identify a sample if the name is unfamiliar.

Merge samples
=============
Select a least two samples from the manager and click the merge button.

Delete samples
==============
Select one or more samples from the manager and click the delete button.

Change main sample
==================
The main sample is the first one in the sample list. The main sample is only necessary when path lengths are
calculated and multiple samples are present [More information here]. To change the main sample, select the
desired sample amd click the change main button





