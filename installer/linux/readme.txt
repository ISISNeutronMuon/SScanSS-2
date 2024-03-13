Before Installation
-------------------
SScanSS 2 requires OpenSSL CA certificates when checking for updates. If you want to
use this feature, install OpenSSL if not installed (Most linux distros have OpenSSL 
pre-installed.) 
	
   > sudo apt-get install openssl

Add environment variable "SSL_CERT_DIR" and set its value to the path of the
CA certificate directory. The OpenSSL install directory can be found by typing

   > openssl version -d
   
The "SSL_CERT_DIR" variable should be set to the path of the "certs" folder in the 
OpenSSL install directory e.g.
	
	SSL_CERT_DIR=/usr/lib/ssl/certs
	
Ensure the variable is permanent, i.e. close and reopen the terminal and check that 
the variable is still available. You may need to write the variable to a file e.g.
"etc/environment" on Ubuntu.


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
