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
echo "Welcome to the SScanSS 2 Installer"
echo ""

INSTALL_EXAMPLES="y"
INSTALL_EDITOR="n"

# Create destination folder
if [[ $EUID -ne 0 ]]; then
    INSTALL_DIR="$HOME/SScanSS-2"
    MENU_DIR="$HOME/.local/share/applications"
    MENU_PATH="$MENU_DIR/sscanss-2.desktop"
    EDITOR_MENU_PATH="$MENU_DIR/sscanss-2-editor.desktop"
    LINK_DIR="$HOME/.local/bin/"
    LINK_PATH="$LINK_DIR/sscanss2"
    USER=$(whoami)
else
    INSTALL_DIR="/usr/local/SScanSS-2"
    MENU_DIR="/usr/share/applications"
    MENU_PATH="$MENU_DIR/sscanss-2.desktop"
    EDITOR_MENU_PATH="$MENU_DIR/sscanss-2-editor.desktop"
    LINK_DIR="/usr/local/bin/"
    LINK_PATH="$LINK_DIR/sscanss2"
    USER=$SUDO_USER
fi
    
# Show License	
more < "./sscanss/LICENSE"

while true
do
  echo ""
  echo "Do you accept all of the terms of the preceding license agreement? (y/n):"
  read -r REPLY
  REPLY=$(echo "$REPLY" | tr '[:upper:]' '[:lower:]')
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
read -r DIR_NAME

if [ "$DIR_NAME" != "" ]; then
    INSTALL_DIR=$DIR_NAME
fi

if [[ -d "$INSTALL_DIR" && "$(ls -A "$INSTALL_DIR")" != "" ]]; then
while true
do
    echo ""
    echo "The destination folder ($INSTALL_DIR) exists. Do you want to remove it? (y/n)"
    read -r REPLY
    REPLY=$(echo "$REPLY" | tr '[:upper:]' '[:lower:]')
    if [ "$REPLY" != y ] && [ "$REPLY" != n ]; then
        echo "        <Please answer y for yes or n for no>" > /dev/tty
    fi
    if [ "$REPLY" = y ]; then
        echo "Removing old installation"

	if ! rm -rf "$INSTALL_DIR"; then
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
    if ! mkdir -p "$INSTALL_DIR"; then
         echo "The $INSTALL_DIR directory does not exist and could not be created."
    exit 1
  fi
fi


echo ""
echo "Install example data used in the tutorials? (y/n) [$INSTALL_EXAMPLES]: "
read -r INSTALL_FLAG
if [ "$INSTALL_FLAG" != "" ]; then
    INSTALL_EXAMPLES=$INSTALL_FLAG
fi

echo ""
echo "Install developer tool for editing instrument description files? (y/n) [$INSTALL_EDITOR]: "
read -r INSTALL_FLAG
if [ "$INSTALL_FLAG" != "" ]; then
    INSTALL_EDITOR=$INSTALL_FLAG
fi

echo ""
echo "Building executable (This should take a few minutes) ..."

python_exec="./envs/sscanss/bin/python3.7"
$python_exec -m pip install --no-cache-dir --no-build-isolation ./packages/* &>/dev/null
if [ "$INSTALL_EDITOR" = y ]; then
    $python_exec "./sscanss/build_executable.py" --skip-tests &>/dev/null
else
    $python_exec "./sscanss/build_executable.py" --skip-tests --skip-editor &>/dev/null
fi
echo "Copying executable and other files ..."

GROUP=$(id -gn "$USER")
cp -ar "./sscanss/installer/bundle/app/." "${INSTALL_DIR}"
if [ "$INSTALL_EXAMPLES" = y ]; then
    cp -ar "./sscanss/examples" "$INSTALL_DIR/examples"
fi
if [ "$INSTALL_EDITOR" = y ]; then
    cp -ar "./sscanss/installer/bundle/editor/." "$INSTALL_DIR/bin"
fi
chown -R "$USER:$GROUP" "$INSTALL_DIR"

# Create Desktop Entry for SScanSS 2
if [ ! -d $MENU_DIR ]; then
    echo "Creating $MENU_DIR"
    mkdir $MENU_DIR 
fi

DESKTOP_ENTRY="[Desktop Entry]
Name=SScanSS 2
Comment=Strain Scanning Simulation Software
Exec=$INSTALL_DIR/bin/sscanss
Icon=$INSTALL_DIR/static/images/logo.png
Type=Application
StartupNotify=true"

if ! echo "$DESKTOP_ENTRY" > $MENU_PATH; then
    echo "Failed to create menu entry for SScanSS 2"
else
    chmod 644 $MENU_PATH
fi

if [ "$INSTALL_EDITOR" = y ]; then
    DESKTOP_ENTRY="[Desktop Entry]
    Name=SScanSS 2 Editor
    Comment=Editor for SScanSS-2 instrument description files
    Exec=$INSTALL_DIR/bin/editor
    Icon=$INSTALL_DIR/static/images/editor-logo.png
    Type=Application
    StartupNotify=true"
    if ! echo "$DESKTOP_ENTRY" > $EDITOR_MENU_PATH; then
        echo "Failed to create menu entry for SScanSS 2 Editor"
    else
        chmod 644 $EDITOR_MENU_PATH
    fi
fi

# Create global link
if [ ! -d $LINK_DIR ]; then
    mkdir $LINK_DIR
fi

ln -sf  "$INSTALL_DIR/bin/sscanss" ${LINK_PATH}
if [ "$INSTALL_EDITOR" = y ]; then
    ln -sf  "$INSTALL_DIR/bin/editor" $LINK_DIR/sscanss2-editor
fi

echo "Installation complete."
echo ""

# Exit from the script with success (0)
exit 0
