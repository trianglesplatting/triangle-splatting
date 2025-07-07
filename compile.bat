@echo off

cd submodules\diff-triangle-rasterization

:: Delete build and egg-info folders if they exist
if exist build rmdir /s /q build
if exist diff_triangle_rasterization.egg-info rmdir /s /q diff_triangle_rasterization.egg-info

pip install .

cd ..\..
