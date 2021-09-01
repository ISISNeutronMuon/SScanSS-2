
[![license](https://img.shields.io/github/license/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/blob/master/LICENSE)
[![release](https://img.shields.io/github/release/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/releases)
[![Actions Status](https://github.com/ISISNeutronMuon/SScanSS-2/workflows/Build/badge.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/actions)
[![Actions Status](https://github.com/ISISNeutronMuon/SScanSS-2/workflows/Docs/badge.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5266561.svg)](https://doi.org/10.5281/zenodo.5266561)

SScanSS 2
=========
SScanSS 2 (pronounced “*scans two*”) provides a virtual laboratory for planning, visualising, and setting-up strain scanning experiments on engineering beam-line instruments.  
SScanSS 2 which is an acronym for **S**train **Scan**ning **S**imulation **S**oftware uses a computer model of the instrument i.e. jaws, collimators, positioning system and 3D model of the sample to simulate the measurement procedure. The main output of the simulation is a script that can be used to drive the real-world positioning system to the desired measurement positions.  
SScanSS 2 is a Python rewrite of the SScanSS application written in IDL by Dr. Jon James at the [Open University](http://www.open.ac.uk), in collaboration with the ISIS Neutron and Muon source. 
In addition to other things, it provides a new UI, improved simulation speed and a more maintainable code. 
  
How to run the code
-------------------
The code is currently known to run on Windows, and Linux; it has not been tested on Mac. Installers are available on the
[release](https://github.com/ISISNeutronMuon/SScanSS-2/releases) page. It should be noted that while the code is 
Python 3 compatible, a single version will be tested for each release to ensure a consistent experience for all users. 
The supported version for the next release is Python 3.7. To run the source:

1. Download the repository
2. Install dependencies
        
        pip install -r requirements.txt
        pip install -r requirements-dev.txt  # optional for development only
3. Add the following line in **main.py** after the ``import sys`` statement  

        import os
        sys.path.append(os.path.abspath('..')) 
4. Open a terminal and navigate to the **SScanSS-2** directory then  
        
        cd sscanss
        python main.py

How to build the documentation
------------------------------
The documentation is currently hosted using Github pages at [https://isisneutronmuon.github.io/SScanSS-2/](https://isisneutronmuon.github.io/SScanSS-2/).
The documentation should be built using the provided Sphinx make file. The restructured text source is in the **docs** folder while 
the build will be placed in the **docs/_build** folder. 

    cd docs
    make clean
    make html

How to build the Installer
--------------------------
### Windows
1. Build the executable using *build_executable.py*. The script will create the executable for the main software in the 
   **installer/bundle** folder and the executable for the instrument editor in the **installer/editor** folder. The 
   instrument editor is a developer tool for creating or modifying instrument description files, using 
   the '--skip-editor' option, will build only the main software.
   
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

Citing SScanSS 2
----------------
1. J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for neutron and synchrotron 
strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, p. 743–746, 2004.

2. Nneji Stephen. (2021). SScanSS 2—a redesigned strain scanning simulation software (Version 1.1.0).
[http://doi.org/10.5281/zenodo.5266561](http://doi.org/10.5281/zenodo.5266561).
