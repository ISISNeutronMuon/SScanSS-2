
[![license](https://img.shields.io/github/license/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/blob/master/LICENSE)
[![release](https://img.shields.io/github/release/ISISNeutronMuon/SScanSS-2.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/releases)
[![Actions Status](https://github.com/ISISNeutronMuon/SScanSS-2/workflows/Build/badge.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/actions)
[![Actions Status](https://github.com/ISISNeutronMuon/SScanSS-2/workflows/Docs/badge.svg)](https://github.com/ISISNeutronMuon/SScanSS-2/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7625691.svg)](https://doi.org/10.5281/zenodo.7625691)

SScanSS 2
=========
SScanSS 2 (pronounced “*scans two*”) provides a virtual laboratory for planning, visualising, and setting-up strain scanning experiments on engineering beam-line instruments.  
SScanSS 2 which is an acronym for **S**train **Scan**ning **S**imulation **S**oftware uses a computer model of the instrument i.e. jaws, collimators, positioning system and 3D model of the sample to simulate the measurement procedure. The main output of the simulation is a script that can be used to drive the real-world positioning system to the desired measurement positions.  
SScanSS 2 is a Python rewrite of the SScanSS application written in IDL by Dr. Jon James at the [Open University](http://www.open.ac.uk), in collaboration with the ISIS Neutron and Muon source. 
In addition to other things, it provides a new UI, improved simulation speed and a more maintainable code. 
  
Installation
------------
The code is currently known to run on Windows, and Linux; it has not been tested on Mac. Installers are available on the
[release](https://github.com/ISISNeutronMuon/SScanSS-2/releases) page. It should be noted that while the code is 
Python 3 compatible, a single version will be tested for each release to ensure a consistent experience for all users. 
The supported version for the next release is Python 3.10.

Citing SScanSS 2
----------------
1. J. A. James, J. R. Santisteban, L. Edwards and M. R. Daymond, “A virtual laboratory for neutron and synchrotron 
strain scanning,” Physica B: Condensed Matter, vol. 350, no. 1-3, p. 743–746, 2004.

2. Nneji Stephen, Sharp Paul, Farooq Rabiya, Zavileiskii Timofei, & Cooper Joshaniel FK. (2022). SScanSS 2—a redesigned 
strain scanning simulation software (Version 2.1.0). [http://doi.org/10.5281/zenodo.7625691](http://doi.org/10.5281/zenodo.7625691).
