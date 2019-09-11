#!/bin/bash

echo ""
echo "Welcome to the SScanSS-2 Installer"
echo ""

BUNDLE="bundle/*"
LICENSE="bundle/LICENSE"
 
# Create destination folder
if [[ $EUID -ne 0 ]]; then
    INSTALL_DIR="$HOME/SScanSS-2"
    MENU_DIR="$HOME/.local/share/applications"
    MENU_PATH="$HOME/.local/share/applications/sscanss-2.desktop"
    LINK_PATH="$HOME/.local/bin/sscanss2"
    LINK_DIR="$HOME/.local/bin/"
else
    INSTALL_DIR="/usr/local/SScanSS-2"
    MENU_DIR="/usr/share/applications"
    MENU_PATH="/usr/share/applications/sscanss-2.desktop"
    LINK_PATH="/usr/local/bin/sscanss2"
    LINK_DIR="/usr/local/bin/"
fi
    

# Show License	
cat ${LICENSE} |more

echo ""
echo "Do you accept all of the terms of the preceding license agreement? (y/n):"
read REPLY
REPLY=`echo $REPLY | tr '[A-Z]' '[a-z]'`
if [ "$REPLY" != y -a "$REPLY" != n ]; then
    echo "        <Please answer y for yes or n for no>" > /dev/tty
fi
if [ "$REPLY" = n ]; then
    echo ">>> Aborting installation"
    exit 1
fi

 
echo ""
echo "Please enter the directory to install in
 
(The default is \"$INSTALL_DIR\")
"
read DIR_NAME

if [ "$DIR_NAME" != "" ]; then
    INSTALL_DIR=$DIR_NAME
fi

if [ -d  "$INSTALL_DIR" ]; then
    echo "The destination folder ($INSTALL_DIR) exists. Do you want to remove it? (y/n)"
    read REPLY
    REPLY=`echo $REPLY | tr '[A-Z]' '[a-z]'`
    if [ "$REPLY" != y -a "$REPLY" != n ]; then
        echo "        <Please answer y for yes or n for no>" > /dev/tty
    fi
    if [ "$REPLY" = y ]; then
        echo ">>> Removing old installation"
	rm -rf $INSTALL_DIR
	if [ $? -ne 0 ]; then
	    echo "Failed to remove old installation"
	    echo "Aborting installation"
	    exit 1
	fi
    else
	    echo "Aborting installation"
	    exit 0
    fi
fi
 
if [ ! -d  "$INSTALL_DIR" ]; then
    mkdir -p $INSTALL_DIR
    if [ $? != 0 ]; then
         echo "
	The $INSTALL_DIR directory does not exist and could not be created.  
	Please create this directory prior to running this script."
    exit 1
    fi
fi

# Find __ARCHIVE__ maker, read archive content and decompress it
echo "Copying Files ..."
echo ""
cp -av ${BUNDLE} ${INSTALL_DIR}

# Create Desktop Entry for SScanSS-2
if [ -f $MENU_DIR ]; then
	echo "Creating $MENU_DIR"
	mkdir $MENU_DIR
fi

echo "[Desktop Entry]
Name=SScanSS-2
Comment=Strain Scanning Simulation Software
Exec=$INSTALL_DIR/bin/sscanss
Icon=$INSTALL_DIR/static/images/logo.png
Type=Application
StartupNotify=true" > $MENU_PATH
if [ $? -ne 0 ]; then
	echo "Failed to create menu entry"
	exit 1
fi

# Create global link
if [ -f $LINK_DIR ]; then
	echo ">>> Creating $LINK_DIR"
	mkdir $LINK_DIR
fi
ln -sf  $INSTALL_DIR/bin/sscanss ${LINK_PATH}

echo ""
echo "Installation complete."
echo ""

# Exit from the script with success (0)
exit 0

