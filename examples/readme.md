# QBB - Quantile-Based Binarization of 3D Point Cloud Descriptors
## Examples

Here you can find examples for the binarizations.

## `lovs_generation.py`
Example to see how the standalone binary descriptors work.
The script will generate a binary descriptor for a part of a point cloud from the `data` folder.

Run with: `python lovs_generation.py`

## `registration.py`
Example to see how to use the binarization methods.
Register two clouds using FPFH and SHOT feature descriptors and their binarized versions B-SHOT, QBB-FPFH.
Then draws the five point clouds in a viewer.

Run with: `python registration.py`