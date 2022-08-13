#!/bin/bash
# This script is used to create development environment.
# It only supports Linux environments
# See intent/multiagents/README.md for more details.
set -e

# Skipping set up aws for public repo
SKIP_AWS=false
if [[ "$1" = '--skip-aws' ]]; then
  SKIP_AWS=true
fi

conda_ver=$(cat conda_ver)
REPO_PATH="$(dirname $(dirname $(realpath $0)))"
echo repo root dir is: $REPO_PATH

# install system dependency for OpenCV
if [[ $( which apt-get ) ]] 2> /dev/null; then
  system_deps='libgl1-mesa-glx git'
  for dep in $system_deps; do
    if ! dpkg -s $dep 2>&1 | grep --quiet "install ok installed" > /dev/null; then
      # only try to install if package is not already installed
      sudo apt-get -y install $dep
    else
      echo Skipping install of $dep
    fi
  done
  if [[ ! $( which aws ) ]] 2> /dev/null; then
    sudo apt-get -y install awscli
  else
    echo Skipping install of awscli
  fi
elif [[ $( which yum ) ]] 2> /dev/null; then
  sudo yum intall mesa-libGL
fi

if [ ! -d ~/miniconda3/ ] && [ ! -d /opt/conda/ ]; then
  # Take action if conda doesn't exists. #
  echo "Miniconda3 not found, installing."
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b
else
  echo Skipping install of miniconda
fi

# Create conda env
cd $REPO_PATH/
if [ -d ~/miniconda3/ ]; then
  . ~/miniconda3/etc/profile.d/conda.sh
elif [ -d /opt/conda/ ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi
conda init


if ! conda env list | grep --quiet "^$conda_ver\s"; then
  conda env create -f environment.$conda_ver.yml
else
  echo Skipping creation of env $conda_ver
fi

# Activate conda env
conda activate $conda_ver

# Install pre-commit
pre-commit install --allow-missing-config
nbdime config-git --enable

if [[ "${SKIP_AWS}" = 'true' ]]; then
  # Skip aws set up for public repo
  echo "Skipping aws set up and data downloading"
  exit 0
fi

# Install local packages
cd $REPO_PATH/

if [ ! -d ~/.aws/ ]; then
  # need AWS credentials
  echo "running aws configure"
  aws configure
fi

# Download data for unit tests from S3
./intent/multiagents/get_unit_test_data.sh
