
[![license](https://img.shields.io/github/license/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/blob/master/LICENSE)
[![release](https://img.shields.io/github/release/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/releases)
[![Actions Status](https://github.com/ISISNeutronMuon/SScanSS-2/workflows/Build/badge.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/actions)

SScanSS 2
=========
SScanSS 2 (pronounced “*scans two*”) provides a virtual laboratory for planning, visualising, and setting-up strain scanning experiments on engineering beam-line instruments.  
SScanSS 2 which is an acronym for **S**train **Scan**ning **S**imulation **S**oftware uses a computer model of the instrument i.e. jaws, collimators, positioning system and 3D model of the sample to simulate the measurement procedure. The main output of the simulation is a script that can be used to drive the real-world positioning system to the desired measurement positions.  
SScanSS 2 is a rewrite of the SScanSS application developed by Dr. Jon James at the [Open University](http://www.open.ac.uk) in collaboration with the ISIS neutron facility, in addition to other things it will provide a new UI, improved simulation speed and a more maintainable code. 
  
How to run the code
-------------------
The code is currently known to run on Windows, and Linux; it has not been tested on Mac. Installers are available on the
[release](https://github.com/ISISNeutronMuon/SScanSS-2/releases) page. To run the source:

1. Download the repository
2. Install dependencies
        
        pip install -r requirements.txt
3. Add the following line in **main.py** after the ``import sys`` statement  

        import os
        sys.path.append(os.path.abspath('..')) 
4. Open a terminal and navigate to the **SScanSS-2** directory then  
        
        cd sscanss
        python main.py

How to build the documentation
------------------------------
The documentation currently hosted using Github pages at [https://isisneutronmuon.github.io/SScanSS-2/](https://isisneutronmuon.github.io/SScanSS-2/).
The source is in **docsrc** folder while the build is in the **docs** folder. Source can be  built using Sphinx make file.

    cd docsrc
    make clean
    make html

How to build the Installer
--------------------------
### Windows
1. Build the executable using *build_executable.py*. The script will create the executable in the **installer/bundle** 
   folder. 
   
       python build_executable.py
    
2. Download and install the [NSIS](https://sourceforge.net/projects/nsis/) application (version 3.04). Open 
   makensisw.exe in the NSIS installation folder, load **installer/windows/build_installer.nsi** into the makensisw 
   compiler. The installer would be created in the **installer/windows** folder.

### Linux
1. The installer can be built by running the **installer/linux/build_installer.sh** bash script. The script requires 
   that [makeself](https://makeself.io/) (version 2.4.0) and git are installed on the machine.

        > ./build_installer.sh --remote --tag v1.0.0

   or
   
        > ./build_installer.sh --local ../.. --tag v1.0.0
        
2. This script will clone SScanSS-2 from the remote or local git repo, download miniconda and required pip packages, then 
   bundle them all into a makeself archive (*.run) which serves as the installer.  The installer would be created in the 
   **installer/linux** folder.

Instrument Editor
-----------------
The instrument editor is a developer tool for creating or modifying instrument description files for SScanSS-2. The 
executable can be built using *build_executable.py* with the '--build-editor' option.

        python build_executable.py --build-editor

Citing SScanSS 2
----------------
1. J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for neutron and synchrotron 
strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, p. 743–746, 2004.
