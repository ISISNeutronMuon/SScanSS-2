Contributing to SScanSS
=======================
Everyone is welcome to contribute to the SScanSS-2 project by either opening an issue (please check that the issue has not been 
reported already) or submitting a pull request.

Create Developer Environment
----------------------------
First begin by creating a fork of the SScanSS-2 repo, then clone the fork

    git clone https://github.com/<username>/SScanSS-2.git
    cd SScanSS-2

We recommend using anaconda python distribution  for development, a new virtual environment should be 
created to isolate dependencies. For Windows, first download and install [Microsoft Visual C++](https://aka.ms/vs/16/release/vc_redist.x64.exe). Then run the following

    conda env create -f environment.yml
    conda activate sscanss

And finally create a separate branch to begin work

    git checkout -b new-feature

Once complete submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) via GitHub. 
Ensure to rebase your branch to include the latest changes on your branch and resolve possible merge conflicts.


Unit-testing and coverage
-------------------------
SScanSS-2 uses the **unittest** module for testing. Proper documentation and unit tests is highly recommended.To run unit 
tests and generate coverage report 

        python make.py --test-coverage

For developers, it is advisable to install a git pre-commit hook to run unit test on each commit. On Windows, you might 
need to add the path of required DLLs e.g. *ANACONDA_DIR/Library/bin* into the system path for the pre-commit hook to work 

        python make.py --add-pre-commit-hook

The pre-commit hook will check that the code is formatted appropriately then run the unit test. To check the code 
formatting without the pre-commit hook

        python make.py --check-code-format

To format the code into the appropriate style

        python make.py --format-code

To run the linter on the code

        python make.py --run-linter

To run the linter, tests, and build the code

        python make.py --build-all

Documentation
-------------
The documentation is currently hosted using GitHub pages at [https://isisneutronmuon.github.io/SScanSS-2/](https://isisneutronmuon.github.io/SScanSS-2/).
The documentation should be built using the provided Sphinx make file. The reStructuredText source is in the **docs** folder while 
the build will be placed in the **docs/_build** folder. 

    cd docs
    make clean
    make html

Style guidelines
----------------
* Docstrings should be written in the reStructuredText format.
* Method names should be camelCase to be consistent with PyQT.
* Function and variable names should be snake_case

How to build the Installer
--------------------------
### Windows
1. To build the executable for SScanSS use,
   
        python make.py --build-sscanss        

   This script will create the executable for the main software only in the **installer/bundle/app** folder. The 
   instrument editor is a developer tool for creating or modifying instrument description files, using the
   '--build-editor' option, will build the executable for the instrument editor in the **installer/bundle/editor** 
   folder. 
   
        python make.py --build-editor    
    
2. Download and install the [NSIS](https://sourceforge.net/projects/nsis/) application (version 3.07). Open 
   makensisw.exe in the NSIS installation folder, load **installer/windows/build_installer.nsi** into the makensisw 
   compiler. The installer would be created in the **installer/windows** folder.

### Linux
1. The installer can be built by running the **installer/linux/build_installer.sh** bash script. The script requires 
   that [makeself](https://makeself.io/) (version 2.4.0) and git are installed on the machine.

        > ./build_installer.sh --remote --tag v1.0.0

   or
   
        > ./build_installer.sh --local ../.. --tag v1.0.0
   
   The script above will clone SScanSS-2 from the remote or local git repo, download miniconda and required pip packages, then 
   bundle them all into a makeself archive (*.run) which serves as the installer. The installer would be created in the 
   **installer/linux** folder.

### Mac Os
1. To build executable for sccanss use:

        > python make.py --build-sscanss

2.  To build executable for editor use:

        > python make.py --build-editor