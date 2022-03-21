"""
Created by Dániel Varga;
"""

import numpy as np
import open3d as o3d

class VBBD_descriptor:
    def __init__(self, g=9, h_mult=4, support_radius=0.1):
        self.g = g
        self.support_radius = support_radius
        self.h = h_mult * support_radius/g
        self.frac = 1/np.sqrt(np.pi*2*self.h)
        self.denom = 2*np.power(self.h,2)
    
    def _cloud_to_lrf(self, cloud_np, lrf_z, lrf_x, lrf_y, p):
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(cloud_np)
        trans_matrix = np.array([[lrf_x[0], lrf_x[1], lrf_x[2], 0],
                                [lrf_y[0], lrf_y[1], lrf_y[2], 0],
                                [lrf_z[0], lrf_z[1], lrf_z[2], 0],
                                [0, 0, 0, 1]])
        temp.translate(p, relative=False)
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(np.asarray(temp.points)-p)
        temp.transform(trans_matrix)
        return temp

    def _get_lrf(self, p_point, local_surface_np):
        leading_weight = 0
        sign_sum = np.zeros(3)
        mat_sum = np.zeros((3,3))

        if len(local_surface_np) == 0:
                raise Exception("Zero neighbors!")

        for q_point in local_surface_np:
            qp_vec = np.subtract(q_point, p_point)
            qp_vec_dist = np.linalg.norm(qp_vec, ord=2)
            radius_minus_len = self.support_radius - qp_vec_dist
            leading_weight += radius_minus_len
            mat_sum += radius_minus_len*np.outer(qp_vec, qp_vec.T)
            sign_sum += qp_vec
        cov_mat = (1/leading_weight)*mat_sum

        e_values, e_vectors = np.linalg.eig(cov_mat)
        min_ind = np.argmin(e_values)
        max_ind = np.argmax(e_values)
        z_axis = e_vectors[:,max_ind]
        x_axis = e_vectors[:,min_ind]

        z_axis = z_axis if np.dot(sign_sum, z_axis) >= 0 else -1*z_axis
        x_axis = x_axis if np.dot(sign_sum, x_axis) >= 0 else -1*x_axis
        y_axis = np.cross(z_axis, x_axis)

        return (x_axis, y_axis, z_axis)

    def _get_result(self, fi_wbd_all):
        fi_wbd_ave = fi_wbd_all.mean()
        result = np.zeros(np.power(self.g, 3), dtype=np.dtype('i1'))
        for i in np.arange(len(result)):
            if fi_wbd_all[i] > fi_wbd_ave:
                result[i] = 1
        return result

    def _get_voxel_centers(self):
        voxel_centers_list = []
        _s = (2*self.support_radius)/self.g
        for i in np.arange(self.g):
            for j in np.arange(self.g):
                for k in np.arange(self.g):
                    x = (i - self.g/2) * _s
                    y = (j - self.g/2) * _s
                    z = (k - self.g/2) * _s
                    voxel_centers_list.append([x, y, z])
        return np.array(voxel_centers_list)

    def _transform_voxel_centers(self, voxel_centers_np, lrf_z, lrf_x, lrf_y, p):
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(voxel_centers_np)
        trans_matrix = np.array([[lrf_x[0], lrf_x[1], lrf_x[2], 0],
                                [lrf_y[0], lrf_y[1], lrf_y[2], 0],
                                [lrf_z[0], lrf_z[1], lrf_z[2], 0],
                                [0, 0, 0, 1]])
        trans_matrix_inv = np.linalg.inv(trans_matrix)
        temp.transform(trans_matrix_inv)
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(np.asarray(temp.points)+p)
        temp.translate(p, relative=False)
        return np.array(temp.points)

    def _get_local_surface(self, p_point, points_np, kdtree):
        [k, neis_idx, _] = kdtree.search_radius_vector_3d(p_point, self.support_radius)
        return points_np[neis_idx], neis_idx


    def _get_wbds(self, lrf_cloud, voxel_centers_np):
        fi_wbd_all = []
        buffer_region_kdtree = o3d.geometry.KDTreeFlann(lrf_cloud)
        lrf_cloud_points_np = np.array(lrf_cloud.points)
        for voxel_center in voxel_centers_np:
            [k, neis_idx, _] = buffer_region_kdtree.search_radius_vector_3d(voxel_center, self.h)
            if k==0:
                fi_wbd_all.append(0)
                continue
            wbd_sum = 0
            for nei in neis_idx:
                t_n = lrf_cloud_points_np[nei]
                numerator = np.power(np.linalg.norm(np.subtract(t_n, voxel_center), ord=2), 2)
                in_sum = self.frac*np.exp(-1*numerator/self.denom)
                wbd_sum += in_sum
            fi_wbd_all.append(wbd_sum/k)
        return np.array(fi_wbd_all)

    def get_feature_for_point(self, points_np, kdtree, point_ind):
        p_point = points_np[point_ind]
        local_surface_np, loc_sur_neis_ind = self._get_local_surface(p_point, points_np, kdtree)
        lrf = self._get_lrf(p_point, local_surface_np)
        lrf_cloud = self._cloud_to_lrf(local_surface_np, lrf[2], lrf[0], lrf[1], p_point)
        voxel_centers_np = self._get_voxel_centers()
        fi_wbd_all = self._get_wbds(lrf_cloud, voxel_centers_np)
        result = self._get_result(fi_wbd_all)
        return result

    def get_feature_for_point_mod(self, points_np, kdtree, point_ind, voxel_centers_np):
        p_point = points_np[point_ind]
        local_surface_np, loc_sur_neis_ind = self._get_local_surface(p_point, points_np, kdtree)
        lrf = self._get_lrf(p_point, local_surface_np)
        lrf_cloud = self._cloud_to_lrf(local_surface_np, lrf[2], lrf[0], lrf[1], p_point)
        fi_wbd_all = self._get_wbds(lrf_cloud, voxel_centers_np)
        result = self._get_result(fi_wbd_all)
        return result

    def get_features(self, full_cloud):
        kdtree = o3d.geometry.KDTreeFlann(full_cloud)
        points_np = np.array(full_cloud.points)
        voxel_centers_np = self._get_voxel_centers()
        features = []
        for point_ind in np.arange(len(points_np)):
            if point_ind % 10000 == 0:
                print(point_ind)
            result = self.get_feature_for_point_mod(points_np, kdtree, point_ind, voxel_centers_np)
            features.append(result)
        return np.array(features)