######################
Simulating Experiments
######################
To run the simulation **Simulation > Run Simulation** and to stop the simulation **Simulation > Stop Simulation**.

.. tip::
   You can start and stop a simulation using keyboard shortcuts **F5** and **Shift+F5** respectively

**************
Quick settings
**************
The **Simulation** menu has a few quick settings which can be applied to each simulation run. Changing the quick settings
will not affect an active simulation, the simulation needs to be restarted for the setting to take effect.

Visualization
=============
Visualization is enabled for simulations by default, it can be toggled by clicking **Simulation > Show Graphically**.
Disabling visualization will increase the speed of the simulation while not significant for a small number of measurements,
simulations with 100 or more measurements may benefit from the speed increase if visualizing the sample position is unimportant.

Hardware Limit
==============
During simulation, positioning system limits are checked by default, this can be toggled by clicking **Simulation > Hardware Limits Check**
When disabled, all the joint limits on the positioning system are ignored (to ignore limit on a single joint see Positioning System).
When enabled, any joint limits that are not explicitly disabled in the positioning system window will be checked.

Collision detection
===================
Collision detection is disabled by default, it can be toggled by clicking **Simulation > Collision Detection**.
When activated, SScanSS-2 will check for collisions at the final sample pose of each measurement and highlight the
colliding bodies in the graphic window (if **Show Graphically** is enabled). The simulation results will also indicate
the point and alignment at which the collision occurred.

.. warning::
    Even though the collision detection in SScanSS-2 is reasonably robust, it should not be a substitute for your eyes
    but a complement. The following should be taken into account:

    1. The software cannot check collisions for objects that are not present such as sample holders, or incomplete models
       of the sample or instrument.
    2. The software only checks for collisions at the final sample pose of a measurement but the path to the pose is not
       checked. It is very possible that the object can collide on its way to the final pose.
    3. Instrument 3D model could differ from real-world because it is a simplification or out of date.

Path Length Calculation
=======================
Path length calculation is disabled by default, it can be toggled by clicking **Simulation > Compute Path Length**.
Path lengths are calculated by checking the distance the beam travels within the sample model. It assumes that the
beam starts outside the sample and every pair of face intersections is taken as beam entry and exit from the sample.
The path length is set to zero if beam hits the gauge volume outside the sample or an entry/exit face pair is not found.

.. warning::
    The path length might be incorrect if the sample has missing faces or spurious faces due to poor scanning
    that intersect with the beam.

The computed path lengths for each measurement will be written into the simulation results and a plot of the path
lengths for each alignment group can be viewed by clicking the plot button.

*************
Export Script
*************
After the simulation is completed, the generated scripts can be exported by clicking **File > Export > Script** or by
clicking the export button on the **Simulation Result** windoow. The **Export Script** dialog wil open, specify the
required microamps for the experiment, and click the export button.

****************
Advanced options
****************
More advanced option for simulation can be accessed by clicking **Simulation > Simulation Options**. The following
options can be changed from the dialog:

* **Execution Order**
* **Position termination tolerance**
* **Orientation termination tolerance**
* **Number of evaluations for global optimization**
* **Number of evaluations for local optimization**