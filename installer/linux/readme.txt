Installation
------------
Run the install script in a terminal as below and follow the prompts to continue 
(Use sudo to install in non-user directory).

   > ./SScanSS-2-installer.run
 
After the installer is completed, run the application by typing the following 

   > sscanss2       

The instrument editor can be installed by typing "y" (yes) when asked by the installer 
to "Install developer tool for editing instrument description files". After installation, 
the instrument editor can be run by typing the following 

   > sscanss2-editor   


Uninstall SScanSS 2
-------------------
To uninstall the SScanSS package, simply delete the installation folder, desktop entry and symbolic link.
If the software is installed with "sudo" the symbolic link and desktop entry will be installed in
"/usr/local/bin/sscanss2" and "/usr/share/applications/sscanss-2.desktop" respectively otherwise
they would be in "$HOME/.local/bin/sscanss2" and "$HOME/.local/share/applications/sscanss-2.desktop".
If the instrument editor is installed, delete its symbolic link also from "usr/local/bin/sscanss2-editor" 
or from "$HOME/.local/bin/sscanss2-editor" for a sudo install.


Troubleshooting
---------------
* If the installer throws the following error

  On Linux, objdump is required. It is typically provided by the 'binutils' package installable via your Linux
  distribution's package manager.

  The solution is to install binutils

  > sudo apt install binutils
