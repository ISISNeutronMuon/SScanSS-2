#!/bin/bash
set -e

command -v gcc >/dev/null 2>&1 || { 
  echo >&2 "gcc is required but not installed.  Aborting."; 
  exit 1; 
}

command -v g++ >/dev/null 2>&1 || { 
  echo >&2 "gcc is required but not installed.  Aborting."; 
  exit 1; 
}

echo ""
echo "Welcome to the SScanSS-2 Installer"
echo ""

# Create destination folder
if [[ $EUID -ne 0 ]]; then
    INSTALL_DIR="$HOME/SScanSS-2"
    MENU_DIR="$HOME/.local/share/applications"
    MENU_PATH="$MENU_DIR/sscanss-2.desktop"
    LINK_DIR="$HOME/.local/bin/"
    LINK_PATH="$LINK_DIR/sscanss2"
    USER=`whoami`
else
    INSTALL_DIR="/usr/local/SScanSS-2"
    MENU_DIR="/usr/share/applications"
    MENU_PATH="$MENU_DIR/sscanss-2.desktop"
    LINK_DIR="/usr/local/bin/"
    LINK_PATH="$LINK_DIR/sscanss2"
    USER=$SUDO_USER
fi
    
# Show License	
cat "./sscanss/LICENSE" |more

while [ 1 ]
do
  echo ""
  echo "Do you accept all of the terms of the preceding license agreement? (y/n):"
  read REPLY
  REPLY=`echo $REPLY | tr '[A-Z]' '[a-z]'`
  if [[ "$REPLY" != y && "$REPLY" != n ]]; then
      echo "        <Please answer y for yes or n for no>" > /dev/tty
  fi
  
  if [ "$REPLY" == y ]; then
      break
  fi
  
  if [ "$REPLY" == n ]; then
      echo "Aborting installation"
      exit 1
  fi
done
 
echo ""
echo "Please enter the directory to install in
 
(The default is \"$INSTALL_DIR\")"
read DIR_NAME

if [ "$DIR_NAME" != "" ]; then
    INSTALL_DIR=$DIR_NAME
fi

if [[ -d "$INSTALL_DIR" && "`ls -A $INSTALL_DIR`" != "" ]]; then
while [ 1 ]
do
    echo ""
    echo "The destination folder ($INSTALL_DIR) exists. Do you want to remove it? (y/n)"
    read REPLY
    REPLY=`echo $REPLY | tr '[A-Z]' '[a-z]'`
    if [ "$REPLY" != y -a "$REPLY" != n ]; then
        echo "        <Please answer y for yes or n for no>" > /dev/tty
    fi
    if [ "$REPLY" = y ]; then
        echo "Removing old installation"
	rm -rf $INSTALL_DIR
	if [ $? -ne 0 ]; then
	    echo "Failed to remove old installation"
	    echo "Aborting installation"
	    exit 1
	fi	
	break
    fi
    if [ "$REPLY" = n ]; then
	echo "Aborting installation"
	exit 0
    fi
done
fi

if [ ! -d  "$INSTALL_DIR" ]; then
    mkdir -p $INSTALL_DIR
    if [ $? != 0 ]; then
         echo "The $INSTALL_DIR directory does not exist and could not be created."
    exit 1
  fi
fi

echo "Building executable (This should take a few minutes) ..."

python_exec="./envs/sscanss/bin/python3.6"
$python_exec -m pip install --no-cache-dir --no-build-isolation ./packages/* &>/dev/null
$python_exec "./sscanss/build_executable.py" --skip-tests &>/dev/null

echo "Copying executable and other files ..."

GROUP=`id -gn $USER`
cp -ar "./sscanss/installer/bundle/." ${INSTALL_DIR}
chown -R $USER:$GROUP $INSTALL_DIR

# Create Desktop Entry for SScanSS-2
if [ ! -d $MENU_DIR ]; then
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
if [ ! -d $LINK_DIR ]; then
    mkdir $LINK_DIR
fi
ln -sf  $INSTALL_DIR/bin/sscanss ${LINK_PATH}

echo "Installation complete."
echo ""

# Exit from the script with success (0)
exit 0

