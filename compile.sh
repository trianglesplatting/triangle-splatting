#!/bin/bash

cd submodules/diff-triangle-rasterization/

# # Delete the build, diff_triangle_rasterization.egg-info, and dist folders if they exist
rm -rf build
rm -rf diff_triangle_rasterization.egg-info

pip install --no-build-isolation .

cd ..
cd ..