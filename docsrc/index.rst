#########################
SScanSS 2's Documentation
#########################

.. image:: images/banner.png
   :alt: banner
   :align: center


SScanSS 2 (pronounced “**scans two**”) provides a virtual laboratory for planning, visualising, and setting-up strain scanning experiments on engineering beam-line instruments.
SScanSS 2 which is an acronym for **S**\train **Scan**\ning **S**\imulation **S**\oftware uses a computer model of the instrument i.e. jaws, collimators, positioning system and 3D
model of the sample to simulate the measurement procedure.The main output of the simulation is a script that can be used to drive the real-world positioning system to
the desired measurement positions.

SScanSS 2 is a rewrite of the SScanSS application developed by Dr. Jon James at the |Open University| in
collaboration with the |isis|.

***************
Using SScanSS 2
***************
SScanSS is based on virtual reality (VR) models of the sample and the instrument. The software manipulates these models
enabling measurement points to be accurately positioned in sample space and simulations of scans to be performed. A
typical use of SScanSS is as follows:

Prior to beam time
==================
1. A model of the sample to be measured is generated, by either: a) LASER scanning the sample, b) exporting a model from a CAD package, or, c) generating a simple model from within SScanSS.
2. The user positions the required measurement points within the sample model.
3. The user specifies the strain components they wish to measure at each measurement point.
4. The virtual instrument is selected and modified to reflect the users choice of (optional) hardware items such as collimators, jaw settings etc.
5. The sample model is positioned within the virtual instrument and the scan is simulated. This simulation is performed in order to:

   * determine how the sample should be oriented in order to measure the required components,
   * determine feasibility, (are all measurement points accessible),
   * estimate count times by checking path lengths, (can the measurement be performed in the available time).

During beam time
================
1. The real sample is positioned on the instrument and its position measured and input into SScanSS.
2. The machine movements required to measure the selected points are generated automatically and the scan is performed.
3. Data is analysed and new measurement points added, as required.

Post beam time
==============
1. The SScanSS software archives all the information required to recreate the experimental setup, thereby assisting in the accurate interpretation of results.
2. Results may be shown in the context of the model for example, by manually overlaying fields on the SScanSS virtual sample model.

************
Installation
************
Installers for Windows and Unix operating system can be download from the project's |release| page. Software
updates will also be accessible from the same page when available. While the software has not been tested on MacOS,
you are welcome to try it, instructions to run the source code are available on |github|.

******
Issues
******
SScanSS 2 is released as a public beta and feedback is necessary to ensure a stable release if you experience
any crashes or unexpected behaviours, do not hesitate to |issues| on the github.

****************
Citing SScanSS 2
****************
1. J.A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for neutron and synchrotron strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, p. 743–746, 2004.

.. |github| raw:: html

   <a href="https://github.com/ISISNeutronMuon/SScanSS-2/" target="_blank">github</a>

.. |release| raw:: html

   <a href="https://github.com/ISISNeutronMuon/SScanSS-2/releases/" target="_blank">release</a>

.. |issues| raw:: html

   <a href="https://github.com/ISISNeutronMuon/SScanSS-2/issues/" target="_blank">open an issue</a>

.. |open university| raw:: html

   <a href="http://www.open.ac.uk/" target="_blank">Open University</a>

.. |isis| raw:: html

   <a href="https://www.isis.stfc.ac.uk/Pages/home.aspx/" target="_blank">ISIS neutron facility</a>
