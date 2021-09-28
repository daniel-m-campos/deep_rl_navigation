#!/bin/bash
#
# Package setup for Ubuntu

REPO=https://github.com/daniel-m-campos/deep_rl.git
SYMLINK_PATH=/usr/local/sbin
declare -A BINARY_DOWNLOAD
BINARY_DOWNLOAD=(
  [Banana.x86_64]=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
  [Reacher.x86_64]=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
  [Tennis.x86_64]=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
)

function cecho() {
  local color=$1
  local exp=$2
  if ! [[ $color =~ ^[0-9]$ ]]; then
    case $(echo "$color" | tr '[:upper:]' '[:lower:]') in
    black | bk) color=0 ;;
    red | r) color=1 ;;
    green | g) color=2 ;;
    yellow | y) color=3 ;;
    blue | b) color=4 ;;
    magenta | m) color=5 ;;
    cyan | c) color=6 ;;
    white | *) color=7 ;;
    esac
  fi
  tput setaf $color
  echo "$exp"
  tput sgr0
}

function err() {
  cecho r "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  exit 1
}

function install_deps() {
  cecho y "Installing dependencies"
  sudo add-apt-repository -y ppa:deadsnakes/ppa &&
    sudo apt-get update -y --allow-unauthenticated &&
    sudo apt install -y python3.6 python3.6-venv python3.6-dev swig
}

function install_package() {
  cecho y "Installing Python package"
  git clone $REPO &&
    cd deep_rl || exit 1 &&
    /usr/bin/python3.6 -m venv venv &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -e .
}

function set_binary_link() {
  BINARY_NAME=$1
  cecho y "Beginning search for $BINARY_NAME"
  BINARY_LOCATION=$(find .. -name "$BINARY_NAME")
  if [[ -z $BINARY_LOCATION ]]; then
    cecho r "Could not find $BINARY_NAME, downloading..."
    URL=${BINARY_DOWNLOAD[$BINARY_NAME]}
    DIR=$(basename "$URL" .zip)
    (wget "$URL" || err "Failed to download $BINARY_NAME") &&
      (unzip -q "$DIR" || err "Failed to unzip $DIR")
    BINARY_LOCATION=$(find "./$DIR" -name "$BINARY_NAME")
  else
    cecho g "Found @ $BINARY_LOCATION"
  fi
  cecho g "Symlinking to $SYMLINK_PATH/$BINARY_NAME"
  ln -s $(realpath $BINARY_LOCATION) "$SYMLINK_PATH/$BINARY_NAME"
}

function set_all_binaries() {
  for BINARY_NAME in ${!BINARY_DOWNLOAD[*]}; do
    set_binary_link "$BINARY_NAME"
  done
}

function main() {
  cecho y "Starting setup of Deep-RL package"
  install_deps &&
    install_package &&
    set_all_binaries
  cecho g "Setup of Deep-RL package complete"
}

main
