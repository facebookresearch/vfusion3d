# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.8.19, CUDA 12.1. Similar package versions might still work but they are not tested.

conda deactivate

# Set environment variables
export ENV_NAME=vfusion3d
export PYTHON_VERSION=3.8.19
export CUDA_VERSION=12.1

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME
conda install pytorch=2.3.0 torchvision==0.18.0 pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
pip install transformers
pip install imageio[ffmpeg]
pip install PyMCubes==0.1.4
pip install trimesh==4.3.2
pip install rembg[gpu,cli]
pip install kiui
