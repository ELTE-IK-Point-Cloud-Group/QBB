"""
Created by Márton Ambrus-Dobai;
"""

import copy
import json
import open3d as o3d
import numpy as np
import sys

sys.path.append('../binarization_methods')
from B_binarization import B_binarization
from QBB_GRAY_binarization import QBB_GRAY_binarization

# Load data
## Load point clouds
cloud_0 = o3d.io.read_point_cloud("./data/clouds/cloud_bin_0.ply")
cloud_1 = o3d.io.read_point_cloud("./data/clouds/cloud_bin_1.ply")
print(':: Point Clouds ✔')
## Load FPFH feature descriptor
fpfh_0 = o3d.pipelines.registration.Feature()
fpfh_1 = o3d.pipelines.registration.Feature()
fpfh_0.data = np.loadtxt("./data/fpfh/cloud_bin_0.csv", delimiter=",").T
fpfh_1.data = np.loadtxt("./data/fpfh/cloud_bin_1.csv", delimiter=",").T
print(':: FPFH Features ✔')
## Load SHOT feature descriptor
shot_0 = o3d.pipelines.registration.Feature()
shot_1 = o3d.pipelines.registration.Feature()
shot_0.data = np.loadtxt("./data/shot/cloud_bin_0.csv", delimiter=",").T
shot_1.data = np.loadtxt("./data/shot/cloud_bin_1.csv", delimiter=",").T
print(':: SHOT Features ✔')

# Voxel size for the dataset
voxel = 0.01

# Binarizations
## B-SHOT
bin = B_binarization()
bshot_0 = o3d.pipelines.registration.Feature()
bshot_1 = o3d.pipelines.registration.Feature()
bshot_0.data = bin.binarize(shot_0.data.T).T
bshot_1.data = bin.binarize(shot_1.data.T).T
print(':: B-SHOT binarization ✔')
## QBB-FPFH
f = open("./data/fpfh/_endpoints.json", "r")
dimension_endpoints = json.load(f)
bin = QBB_GRAY_binarization(dimension_endpoints)
bfpfh_0 = o3d.pipelines.registration.Feature()
bfpfh_1 = o3d.pipelines.registration.Feature()
bfpfh_0.data = bin.binarize(fpfh_0.data.T).T
bfpfh_1.data = bin.binarize(fpfh_1.data.T).T
print(':: QBB-FPFH binarization ✔')

# Registration
def execute_global_registration(source, target, source_feature,
                                target_feature, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99))
    return result
fpfh_result = execute_global_registration(cloud_0, cloud_1, fpfh_0, fpfh_1, voxel)
print(':: FPFH registration ✔')
shot_result = execute_global_registration(cloud_0, cloud_1, shot_0, shot_1, voxel)
print(':: SHOT registration ✔')
bfpfh_result = execute_global_registration(cloud_0, cloud_1, bfpfh_0, bfpfh_1, voxel)
print(':: BFPFH registration ✔')
bshot_result = execute_global_registration(cloud_0, cloud_1, bshot_0, bshot_1, voxel)
print(':: BSHOT registration ✔')



# Visualize results
## Create clouds
source_fpfh = copy.deepcopy(cloud_0)
source_shot = copy.deepcopy(cloud_0)
source_bfpfh = copy.deepcopy(cloud_0)
source_bshot = copy.deepcopy(cloud_0)
target = copy.deepcopy(cloud_1)

## Color clouds
source_fpfh.paint_uniform_color([1, 0.706, 0])
source_shot.paint_uniform_color([0, 0.651, 0.929])
source_bfpfh.paint_uniform_color([0.8, 0.506, 0])
source_bshot.paint_uniform_color([0, 0.451, 0.729])
target.paint_uniform_color([0, 0, 0])

## Transorm the the source clouds into the target cloud
source_fpfh.transform(fpfh_result.transformation)
source_shot.transform(shot_result.transformation)
source_bfpfh.transform(bfpfh_result.transformation)
source_bshot.transform(bshot_result.transformation)

## Visualize the clouds
o3d.visualization.draw_geometries([
        source_fpfh,
        source_shot,
        source_bfpfh,
        source_bshot,
        target],
    zoom=0.75,
    front=[0.12, 0.07, -0.98],
    lookat=[-0.29, -0.24, 3.76],
    up=[0.08, -0.99, -0.06])
