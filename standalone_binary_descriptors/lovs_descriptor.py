"""
Created by Dániel Varga;
"""

import numpy as np
import open3d as o3d

class LOVS_descriptor:
    def __init__(self, m = 7, radius = 0.1):
        self.m = m
        self.radius = radius

    def _get_z_axis(self, cloud, point_ind):
        return np.asarray(cloud.normals)[point_ind]

    def _get_border_points(self, qs_cloud, point):
        kdtree = o3d.geometry.KDTreeFlann(qs_cloud)
        [k1, big_radius_idx, _] = kdtree.search_radius_vector_3d(point, self.radius)
        [k2, small_radius_idx, _] = kdtree.search_radius_vector_3d(point, 0.85*self.radius)
        diff = k1-k2
        border_idx = np.setdiff1d(big_radius_idx,small_radius_idx)
        if diff != len(border_idx):
            raise Exception("KNN search error!")
        return border_idx

    def _get_projected_point(self, qs_cloud, p, p_normal, q):
        pq = np.subtract(q, p) # vector from p to q
        dist = np.dot(pq, p_normal)
        projected_point = np.subtract(pq, p_normal*dist)
        return projected_point, dist**2 # dist**2 will be the weight what we will need

    def _get_x_axis(self, qs_cloud, point, point_normal):
        border_points_idx = self._get_border_points(qs_cloud, point)
        if len(border_points_idx) == 0:
            return [0.0, 0.0, 0.0]
        vector_sum = np.array([0.0, 0.0, 0.0])
        for border_idx in border_points_idx:
            q = np.asarray(qs_cloud.points)[border_idx]
            projected_point, weight = self._get_projected_point(qs_cloud, point, point_normal, q)
            vector_sum = np.add(vector_sum, projected_point * weight)
        divide = np.linalg.norm(vector_sum, ord=2)
        if divide == 0:
            return [0.0, 0.0, 0.0]
        return vector_sum/divide

    def _get_qs_cloud(self, points, normals, kdtree, point):
        
        [k, idx, _] = kdtree.search_radius_vector_3d(point, np.sqrt(3)*self.radius)
        
        out_cloud = o3d.geometry.PointCloud()    
        out_cloud.points = o3d.cpu.pybind.utility.Vector3dVector(points[idx])
        out_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(normals[idx])
        
        return out_cloud

    def _cloud_to_lrf(self, cloud, lrf_z, lrf_x, lrf_y, p):
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(np.asarray(cloud.points))
        trans_matrix = np.array([[lrf_x[0], lrf_x[1], lrf_x[2], 0],
                                [lrf_y[0], lrf_y[1], lrf_y[2], 0],
                                [lrf_z[0], lrf_z[1], lrf_z[2], 0],
                                [0, 0, 0, 1]])
        temp.transform(trans_matrix)
        temp.translate(p, relative=False)
        temp.points = o3d.cpu.pybind.utility.Vector3dVector(np.asarray(temp.points)-p)
        return temp

    def _get_voxel_ind(self, qc_cloud, point_ind):
        l_step = (2*self.radius)/self.m
        temp_point = np.asarray(qc_cloud.points)[point_ind]
        fst = np.floor((temp_point[2]+self.radius)/l_step)*(self.m**2)
        snd = np.floor((temp_point[1]+self.radius)/l_step)*self.m
        thr = np.floor((temp_point[0]+self.radius)/l_step)
        voxel_id = fst + snd + thr 
        return int(voxel_id)
    
    def _get_voxel_ind2(self, qc_cloud, point_ind):
        l_step = (2*self.radius)/self.m
        temp_point = np.asarray(qc_cloud.points)[point_ind]
        fst = np.floor((temp_point[2]+self.radius)/l_step)
        snd = np.floor((temp_point[1]+self.radius)/l_step)
        thr = np.floor((temp_point[0]+self.radius)/l_step)
        return (fst, snd, thr)

    def _get_qc_cloud(self, cloud, p):
        cubic_cloud = o3d.geometry.PointCloud()
        cubic_points = []
        for point in np.asarray(cloud.points):
            if np.abs(point[0]) < self.radius and np.abs(point[1]) < self.radius and np.abs(point[2]) < self.radius:
                cubic_points.append(point)
        cubic_cloud.points = o3d.cpu.pybind.utility.Vector3dVector(np.array(cubic_points))
        return cubic_cloud

    def get_feature_for_point_old(self, points, normals, ind, kdtree):
        
        point = points[ind]
        point_normal = normals[ind]
        
        qs_cloud = self._get_qs_cloud(points, normals, kdtree, point)
        
        lrf_z = point_normal
        lrf_x = self._get_x_axis(qs_cloud, point, point_normal)
        if lrf_x == [0.0, 0.0, 0.0]:
            return np.zeros(self.m**3, dtype=np.int8)
        lrf_y = np.cross(lrf_z, lrf_x)
        
        qs_cloud_lrf = self._cloud_to_lrf(qs_cloud, lrf_z, lrf_x, lrf_y, point)

        qc_cloud = self._get_qc_cloud(qs_cloud_lrf, point)

        voxels = {}

        for i in np.arange(self.m**3):
            voxels[i] = 0

        for qc_point_ind in np.arange(len(np.asarray(qc_cloud.points))):
            voxel_id = self._get_voxel_ind(qc_cloud, qc_point_ind)
            if(voxel_id < 0):
                print(qc_point_ind)           
            voxels[voxel_id] += 1

        result = np.zeros(self.m**3, dtype=np.int8)
        for voxel_ind in np.arange(self.m**3):
            result[voxel_ind] = 1 if voxels[voxel_ind] > 0 else 0
        return result
    
    def _init_grid(self, grid, qc_cloud):
        for qc_point_ind in np.arange(len(np.asarray(qc_cloud.points))):
            voxel_inds = self._get_voxel_ind2(qc_cloud, qc_point_ind)
            grid[int(voxel_inds[0])][int(voxel_inds[1])][int(voxel_inds[2])] = 1
        return grid
        
    def _get_neis(self, grid, ind):
        neis = grid[max(1, ind[0]-1):min(ind[0]+2, self.m-1),
                   max(1, ind[1]-1):min(ind[1]+2, self.m-1),
                   max(1, ind[2]-1):min(ind[2]+2, self.m-1)]
        return neis
    
    def get_feature_for_point(self, points, normals, ind, kdtree, connectivity=False):
        
        point = points[ind]
        point_normal = normals[ind]
        
        qs_cloud = self._get_qs_cloud(points, normals, kdtree, point)
        
        lrf_z = point_normal
        lrf_x = self._get_x_axis(qs_cloud, point, point_normal)
        lrf_y = np.cross(lrf_z, lrf_x)
        
        qs_cloud_lrf = self._cloud_to_lrf(qs_cloud, lrf_z, lrf_x, lrf_y, point)

        qc_cloud = self._get_qc_cloud(qs_cloud_lrf, point)
        
        grid = np.zeros((self.m,self.m,self.m)) # z, y, x
        grid = self._init_grid(grid, qc_cloud)
        
        result = np.zeros(self.m**3, dtype=np.int8)
        for z in np.arange(self.m):
            for y in np.arange(self.m):
                for x in np.arange(self.m):
                    res_ind = z*(self.m**2)+y*self.m+x
                    if connectivity:
                        if grid[z][y][x] > 0:
                            neis = self._get_neis(grid, (z,y,x))
                            if neis.sum() == 1:
                                result[res_ind] = 0
                            else:
                                result[res_ind] = 1
                        else:
                            result[res_ind] = 0
                    else:
                        if grid[z][y][x] > 0:
                            result[res_ind] = 1
                        else:
                            result[res_ind] = 0
                        
        return result
    
    def get_features(self, full_cloud, connectivity=False):
        kdtree = o3d.geometry.KDTreeFlann(full_cloud)
        points = np.asarray(full_cloud.points)
        normals = np.asarray(full_cloud.normals)
        all_feature = []
        for i in np.arange(len(points)):
            feature = self.get_feature_for_point(points, normals, i, kdtree, connectivity)
            all_feature.append(feature)
        return np.array(all_feature)