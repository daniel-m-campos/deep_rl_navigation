#!/bin/bash
#
# Package setup for Ubuntu

BINARY_NAME=Banana.x86_64
BINARY_LINK=/usr/local/sbin/Banana.x86_64
BINARY_DOWNLOAD=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip

function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  exit 1
}

function install_deps() {
  echo Installing dependencies
  sudo add-apt-repository -y ppa:deadsnakes/ppa &&
    sudo apt update -y &&
    sudo apt install -y python3.6 python3.6-venv python3.6-dev swig
  return 0
}

function install_package() {
  echo Installing Python package
  git clone https://github.com/daniel-m-campos/deep_rl_navigation.git &&
    cd deep_rl_navigation || exit 1 &&
    /usr/bin/python3.6 -m venv venv &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -e .
  return 0
}

function set_binary_link() {
  echo Begining search for $BINARY_NAME
  BINARY_LOCATION=$(find / -name $BINARY_NAME)
  if [[ -z $BINARY_LOCATION ]]; then
    echo "Could not find $BINARY_NAME, downloading..."
    wget $BINARY_DOWNLOAD && unzip Banana_Linux | err "Failed to download and unzip $BINARY_NAME"
    BINARY_LOCATION=$(find ./Banana_Linux -name $BINARY_NAME)
  else
    echo Found @ "$BINARY_LOCATION"
  fi
  echo Symlinking to $BINARY_LINK
  ln -s "$BINARY_LOCATION" $BINARY_LINK
  return 0
}

function main() {
  install_deps &&
    install_package &&
    set_binary_link
}

main
