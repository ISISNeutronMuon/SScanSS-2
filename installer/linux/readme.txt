Before Installation
-------------------------------------------
The SScanSS-2 installer requires gcc, g++, which can  be accessed via the OS package
manager for example, this can be installed on Ubuntu by typing the command below in
a terminal.
   
   > sudo apt-get install gcc g++
   
SScanSS-2 requires OpenSSL CA certificates when checking for updates. If you want to 
use this feature, install OpenSSL if not installed (Most linux distros have OpenSSL 
pre-installed.) 
	
   > sudo apt-get openssl

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
-------------------------------------------
Run the install script in a terminal as below (Use sudo to install in non-user directory).

   > ./SScanS-2-installer.run
   
Run the application by typing the following 

   > sscanss2       
   

Uninstall SScanSS-2
-------------------------------------------
To uninstall the SScanSS package, simply delete the installation folder, desktop entry and symbolic link.
If the software is installed with "sudo" the symbolic link and desktop entry will be installed in
"/usr/local/bin/sscanss2" and "/usr/share/applications/sscanss-2.desktop" respectively otherwise
they would be in "$HOME/.local/bin/sscanss2" and "$HOME/.local/share/applications/sscanss-2.desktop"
