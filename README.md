SScanSS 2
=================
SScanSS 2 (pronounced “*scans two*”) provides a virtual laboratory for planning, visualising, and setting-up strain scanning experiments on engineering beam-line instruments.  
SScanSS 2 which is an acronym for **S**train **Scan**ning **S**imulation **S**oftware uses a computer model of the instrument i.e. jaws, collimators, positioning system and 3D model of the sample to simulate the measurement procedure. The main output of the simulation is a script that can be used to drive the real-world positioning system to the desired measurement positions.  
SScanSS 2 is a rewrite of the SScanSS application developed by Dr. Jon James at the [Open University](http://www.open.ac.uk) in collaboration with the ISIS neutron facility, in addition to other things it will provide a new UI, improved simulation speed and a more maintainable code. 

Requirements
------------
This will be moved to requirements.txt later
  * python 3.6
  * PyQt5
  * NumPy
  * h5py
  * pyopengl
  
How to run the code
--------------------
The code is currently known to run on Windows, and Linux; it has not been tested on Mac.  
When the first release is complete, an installer will be provided but this information is for curious minds.

1. Download the repository 
2. Add the following line in **main.py** after the ``import sys`` statement  
    ``sys.path.append('path to the SScanSS-2 folder')``
3. Open a terminal and navigate to the **SScanSS-2** directory then  
    ``cd sscanss``  
    ``python main.py`` 

Citing SScanSS 2
----------------
1. J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for neutron and synchrotron strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, p. 743–746, 2004. 