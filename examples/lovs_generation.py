"""
Created by Dániel Varga;
"""

import numpy as np
import open3d as o3d
import sys

sys.path.append('../standalone_binary_descriptors')
from lovs_descriptor import *

def main():
    print("-------------------------------- START! --------------------------------")

    full_cloud = o3d.io.read_point_cloud("./data/clouds/cloud_bin_0.ply")
    full_cloud.estimate_normals()

    # smaller cloud from the first thousand points
    points = np.asarray(full_cloud.points)[:1000]
    normals = np.asarray(full_cloud.normals)[:1000]
    part_cloud = o3d.geometry.PointCloud()
    part_cloud.points = o3d.cpu.pybind.utility.Vector3dVector(points)
    part_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(normals)
    kdtree = o3d.geometry.KDTreeFlann(part_cloud)

    # number of voxels: m^3, feature calculation radius: radius
    lovs_descriptor_generator = LOVS_descriptor(m=7, radius=0.1)

    # generate feature for one point (third parameter is the index of the point)
    descriptor = lovs_descriptor_generator.get_feature_for_point(points, normals, 0, kdtree, connectivity=False)

    # feature descriptor generation for the whole cloud
    all_descriptor = lovs_descriptor_generator.get_features(part_cloud)
    print(all_descriptor.shape)

    print("--------------------------------- END! ---------------------------------")


if __name__ == "__main__":
    main()