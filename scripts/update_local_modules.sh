#!/bin/bash
set -ex

conda_ver=$(cat conda_ver)

pushd radutils/ && conda run -n $conda_ver pip install -e . && popd
pushd triceps/ && conda run -n $conda_ver pip install -e . && popd
pushd tristan/ && conda run -n $conda_ver pip install -e . && popd
conda run -n $conda_ver pip install -e .
