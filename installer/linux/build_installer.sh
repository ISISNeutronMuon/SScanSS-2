#!/bin/bash
set -e

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -t|--tag)
    TAG="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--local)
    LOCAL=$(realpath "$2")
    shift # past argument
    shift # past value
    ;;
    -d|--build-dir)
    BUILD_DIR=$(realpath "$2")
    shift # past argument
    shift # past value
    ;;
    -r|--remote)
    REMOTE=YES
    shift # past argument
    ;;
    -h|--help)
    HELP=YES
    shift # past argument
    ;;
    *)    # unknown option
    echo >&2 "error: unrecognized command line option '$key'"
    shift
    ;;
esac
done

if [ -n "$HELP" ]; then
  echo "Usage: build_installer [options]"
  echo "Options:"
  echo "-h, --help		     Show this help message and exit"
  echo "-l <dir>, --local <dir>	     Clone SScanSS 2 from local directory (requires git)"
  echo "-d <dir>, --build-dir <dir>  Specify build directory (temp directory will be used if not provided)"
  echo "-t <arg>, --tag <arg>	     Clone specific tag of SScanSS 2 from local (requires git) or web"
  echo "-r, --remote		     Clone SScanSS 2 from Github repo"
  exit 0
fi


command -v makeself >/dev/null 2>&1 || {
  echo >&2 "makeself is required but not installed";
  exit 1;
}

# trap ctrl-c and call finish()
trap finish INT EXIT

function finish() {
  [ -n "$TMP_DIR" ] && rm -rf "$TMP_DIR"
}

SRC_DIR=$(dirname "$(realpath "$0")")

echo ""
echo "SScanSS 2 Installer Builder"
echo ""

if [ -n "$BUILD_DIR" ]; then
  if [ "$(ls -A $BUILD_DIR)" ]; then
    echo >&2 "error: the build directory '$BUILD_DIR' must be empty"
    exit 1
  else
    TMP_DIR="$BUILD_DIR/build"
    mkdir "$TMP_DIR"
  fi
else
  TMP_DIR=$(mktemp -d)
fi
cd "$TMP_DIR"
mkdir "$TMP_DIR/sscanss"

if [ -n "$REMOTE" ]; then
  echo "Downloading SScanSS 2 from remote repo"
  if [ -n "$TAG" ]; then
    SSCANSS_URL="https://github.com/ISISNeutronMuon/SScanSS-2/archive/${TAG}.tar.gz"
  else
    SSCANSS_URL="https://github.com/ISISNeutronMuon/SScanSS-2/tarball/master"
  fi

  wget $SSCANSS_URL -O "$TMP_DIR/sscanss.tar.gz"
  tar xzf "$TMP_DIR/sscanss.tar.gz" -C "$TMP_DIR/sscanss" --strip-components=1

elif [ -n "$LOCAL" ]; then
  command -v git >/dev/null 2>&1 || {
    echo >&2 "git is required with the --local option.";
    exit 1;
  }
  echo "Cloning SScanSS 2 from local directory"
  if [ -n "$TAG" ]; then
    git clone --branch "$TAG" "$LOCAL" "$TMP_DIR"/sscanss
  else
    git clone "$LOCAL" "$TMP_DIR"/sscanss
  fi
else
  echo >&2 "error: location of SScanSS 2 directory is not specified"
  echo "use --help option to see available commands"
  exit 1
fi


echo ""
echo "Downloading Miniconda"
echo ""
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_DIR/miniconda.sh"
bash ./miniconda.sh -b -p ./miniconda

./miniconda/bin/conda create -n sscanss -y python=3.10

echo ""
echo "Downloading Dependencies"
echo ""
python_exec="./miniconda/envs/sscanss/bin/python"
mkdir "$TMP_DIR/packages"
$python_exec -m pip download -r "./sscanss/requirements.txt" --dest "$TMP_DIR/packages"
# workaround for centos 7
$python_exec -m pip download --only-binary=":all:" --platform="manylinux_2_17_x86_64"  --dest "$TMP_DIR/packages" pillow==9.2

echo ""
echo "Compressing Package.tar.gz ..."
echo ""

STAGE_DIR="$TMP_DIR/stage"
mkdir "$STAGE_DIR"

mv -t "$STAGE_DIR" "sscanss" "miniconda/envs" "packages"
chmod 777 "$STAGE_DIR/sscanss/installer/linux/install.sh"

echo ""
echo "Creating self-extracting archive"
echo ""
EXECUTABLE="SScanSS-2-installer.run"
if [ -n "$TAG" ]; then
  EXECUTABLE="SScanSS-2-${TAG:1}-installer.run"
fi

makeself --tar-format "posix" --tar-extra "--exclude=.git --exclude=sscanss/3rdparty --exclude=sscanss/docsrc --exclude=sscanss/docs" "$STAGE_DIR" "$EXECUTABLE" "SScanSS-2 installer" ./sscanss/installer/linux/install.sh
cp -a "$EXECUTABLE" "$SRC_DIR/$EXECUTABLE"

exit 0
