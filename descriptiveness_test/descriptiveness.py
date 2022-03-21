"""
Created by Dániel Varga;
"""

from utils import *
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from itertools import islice
from gmpy2 import mpz


class DescriptivenessEvaluator:

    def __init__(self, input_data):
        self.dim = input_data["dim"]
        self.descriptor = input_data["descriptor"]
        self.cloud_pairs = input_data["cloud_pairs"]
        self.dataset_path = input_data["dataset_path"]
        self.dataset_name = input_data["dataset_name"]
        self.repeat = input_data["repeat"]
        self.sampleNum = input_data["sampleNum"]
        self.keypoint_support_radius = input_data["keypoint_support_radius"]
        self.input_cloud_format = input_data["input_cloud_format"]
        self.is_binary = input_data["is_binary"]
        self.persistence_file = input_data["persistence_file"]
        self.read_persisted_data = input_data["read_persisted_data"]
        self.use_custom_metric = input_data["use_custom_metric"]

    # Repeats the evaluation N times, calculate the mean of recall and precision values
    def repeadtedEvaluation(self, cloud1, cloud2, features1, features2):
        samplePrec = []
        sampleRec = []
        sampleAuc = []
        recMean = []
        precMean = []

        # Repeat the evaluation N times
        for k in np.arange(0, self.repeat):
            print("Repeat: ", k)
            (precisions, recalls, auc) = self.evaluate(cloud1, cloud2, features1, features2)
            sampleRec.append(np.array(recalls))
            samplePrec.append(np.array(precisions))
            sampleAuc.append(auc)

        sampleRec = np.array(sampleRec, dtype='object')
        samplePrec = np.array(samplePrec, dtype='object')

        # Calculate the mean recall and precision values then return
        for n in np.arange(0, len(sampleRec[0])):
            recMean.append(np.mean([row[n] for row in sampleRec]))
            precMean.append(np.mean([row[n] for row in samplePrec]))
        return (recMean, precMean, sampleAuc)

    # Samples keypoints, calculates the number of ground truth correspondences
    def evaluate(self, cloud1, cloud2, features1, features2):
        # Sample keypoints (# sampleNum)
        keypoint_ind1 = np.random.choice(len(features1), self.sampleNum, replace=False)
        keypoint_ind2 = np.random.choice(len(features2), self.sampleNum, replace=False)

        # Create kd-trees for the feature space of the keypoints feature descriptors
        tree1 = None
        tree2 = None
        if(self.is_binary):
            pass
        else:
            tree2 = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm='ball_tree', n_jobs=1)
            tree2.fit(features2[keypoint_ind2])

        # kd-trees for Euclidean space
        tree1_euc = None
        tree2_euc = KDTree(np.asarray(cloud2.points)[keypoint_ind2], leaf_size=40)

        # Get the number of ground truth correspondences
        (gt_corrs, gt_matches) =  self.getGroundTruth(cloud1, cloud2, keypoint_ind1, keypoint_ind2,
                                                 tree1, tree2, tree1_euc, tree2_euc)
        # Get precision and recall values
        (precisions, recalls) = self.getPrecisionRecall(cloud1, cloud2, features1, features2, keypoint_ind1, keypoint_ind2,
                                                   tree1, tree2, tree1_euc, tree2_euc,
                                                   gt_corrs, gt_matches)
        # Calculate the AUC value
        auc = np.trapz(precisions, recalls)
        #visualizeCorrespondences(cloud1, cloud2, gt_corrs)
        return (precisions, recalls, auc)

    # Calculates the precision and recall values
    def getPrecisionRecall(self, cloud1, cloud2, features1, features2, keypoint_ind1, keypoint_ind2, tree1, tree2, tree1_euc, tree2_euc, gt_corrs, gt_matches):
        precisions = []
        recalls = []
        dist_and_corr = []
        # Find correspondences basedon nearest neighbor search
        
        if(self.use_custom_metric):
            # 481 mersenne
            # 77 gray
            mtx1 = np.array([to_arr(x, self.dim) for x in features1[keypoint_ind1]])
            mtx2 = np.array([to_arr(x, self.dim) for x in features2[keypoint_ind2]])
            

        if(self.use_custom_metric):
            inds = np.arange(len(keypoint_ind1))
        else:
            inds = keypoint_ind1

        for ind1 in inds:
            dist = None
            ind2 = None
            if(self.is_binary):
                if(self.use_custom_metric):
                    dist, ind2 = my_nn_custom(mtx1[ind1], mtx2)
                else:
                    dist, ind2 = my_nn(features1[ind1], features2[keypoint_ind2])
            else:
                dist, ind2 = tree2.kneighbors(features1[ind1].reshape(1, -1), return_distance=True)

            if dist[0][1] != 0:
                # Check ratio of first and second nearest neighbor
                dist_ratio = dist[0][0]/dist[0][1]
            else:
                dist_ratio = 0
            
            if(self.use_custom_metric):
                corr_pair = (keypoint_ind1[ind1], keypoint_ind2[ind2[0][0]])
            else:
                corr_pair = (ind1, keypoint_ind2[ind2[0][0]])

            dist_and_corr.append([dist_ratio, corr_pair])

        # Iterate over an interval
        for tau in np.arange(0.5, 1.01, 0.1, dtype=float):
            corr1 = []
            
            for i in np.arange(len(keypoint_ind1)):
                # Add to correspondences if the ratio smaller than tau
                if(dist_and_corr[i][0] < tau):
                    corr1.append(dist_and_corr[i][1])

            # Reciprocity test for correspondences. Increases the possibility of correct correspondences, decreases the number of correspondences.
            # corr2 = []
            # for ind2 in keypoint_ind2:
            #    dist, ind1 = tree1.query(features2[ind2].reshape(1, -1), k=3)
            #    if dist[0][1] != 0:
            #        dist_ratio = dist[0][0]/dist[0][1]
            #        if(dist_ratio < tau):
            #            corr2.append((keypoint_ind1[ind1[0][0]], ind2))

            #corrs = intersect2D(np.array(corr1), np.array(corr2))
            corrs = corr1

            # Number of found correspondences
            matches = len(corrs)

            correct_matches = 0

            for (ind1, ind2) in corrs:
                # Get the distance betwwen th two points of the correspondence
                d = np.linalg.norm(np.asarray(cloud2.points)[ind2] - np.asarray(cloud1.points)[ind1], ord=2)
                # Check that their distance is smaller than the support radius
                if d < self.keypoint_support_radius:
                    correct_matches = correct_matches + 1

            if matches == 0 or gt_matches == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            # Calculate the precision and recall values
            precision = correct_matches/matches
            recall = correct_matches/gt_matches

            precisions.append(precision)
            recalls.append(recall)

        return (precisions, recalls)

    # Search for possible correct corresnponences and count them
    def getGroundTruth(self, cloud1, cloud2, keypoint_ind1, keypoint_ind2, tree1, tree2, tree1_euc, tree2_euc):
        gt_corrs = []
        # Iterate over first cloud keypoints
        for ind1 in keypoint_ind1:
            # Get its nearest neighbor from the other cloud
            dist, ind2 = tree2_euc.query(np.asarray(cloud1.points)[ind1].reshape(1, -1), k=1)
            # If its nearest neighbor closer than the support radius, these two points can form a correct correspondence
            if dist[0][0] < self.keypoint_support_radius:
                gt_corrs.append((ind1, keypoint_ind2[ind2[0]]))
        # Count how many correct correspondence is possible according to the actual current keypoints
        gt_matches = len(gt_corrs)
        return (gt_corrs, gt_matches)

    # Reads cloud pairs and their features. Runs the evaluation and returns with recall, precision and auc values.
    def get_auc_pairs(self):
        first_pair = -1
        auc_pairs = []
        auc_pairs_2 = []
        recList = []
        precList = []

        # Read the ground turth transformations
        f = open(self.dataset_path + self.dataset_name + "/gt.log", "r")

        # Read persisted data
        if(self.read_persisted_data):
            save = open("save_rops_mersenne.data", "r")
            first_pair = int(save.readline())
            auc_pairs = list(map(float, save.readline()[:-2].split('|')))
            auc_pairs_2 = list(map(lambda l: list(map(float, l[1:-2].split(', '))), save.readline()[:-2].split('|')))
            recList = list(map(lambda l: list(map(float, l[1:-2].split(', '))), save.readline()[:-2].split('|')))
            precList = list(map(lambda l: list(map(float, l[1:-2].split(', '))), save.readline()[:-2].split('|')))
            save.close()

        # Run evaluation for every cloud pair
        for pairnum in np.arange(0, self.cloud_pairs):
            line = f.readline()
            if not line:
                break
            if pairnum <= first_pair:
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                continue

            # File id-s for the current clouds
            file1, file2, _ =  map(int, line.split(' '))
            print('File numbers:' + str(file1) + ", " + str(file2) )
            print('Pair number:' + str(pairnum))

            lines = islice(f, 0, 4)
            linesTrans = list(lines)
            gt_trans = np.genfromtxt(linesTrans)

            name1 = self.dataset_path + self.dataset_name + "/clouds/cloud_bin_" + str(file1) + self.input_cloud_format
            name2 = self.dataset_path + self.dataset_name + "/clouds/cloud_bin_" + str(file2) + self.input_cloud_format

            # Read the clouds and align them with ground truth transformation
            cloud1 = o3d.io.read_point_cloud(name1)
            cloud2 = o3d.io.read_point_cloud(name2)
            cloud2.transform(gt_trans)

            fname1 = ''
            fname2 = ''

            fname1 = self.dataset_path + self.dataset_name + "/" + self.descriptor + "/cloud_bin_" + str(file1) + ".csv"
            fname2 = self.dataset_path + self.dataset_name + "/" + self.descriptor + "/cloud_bin_" + str(file2) + ".csv"                

            features1 = None
            features2 = None
            # Read the feature descriptors for the clouds
            if(self.is_binary):
                features1_temp = np.loadtxt(fname1, delimiter=",", dtype=np.dtype('str'))
                features2_temp = np.loadtxt(fname2, delimiter=",", dtype=np.dtype('str'))
                features1 = np.array([mpz(x) for x in features1_temp])
                features2 = np.array([mpz(x) for x in features2_temp])
            else:
                skiprows_num = 0
                features1 = np.loadtxt(fname1, skiprows=skiprows_num, delimiter=",")
                features2 = np.loadtxt(fname2, skiprows=skiprows_num, delimiter=",")
            print("Data loading finnished...")

            # Start the evaluation for the current cloud pair
            (recMean, precMean, sampleAuc) = self.repeadtedEvaluation(cloud1, cloud2, features1, features2)
            # Calculate the AUC value
            auc_pr = np.trapz(precMean, recMean)
            auc_pairs.append(auc_pr)
            auc_pairs_2.append(sampleAuc)
            # Append the recall and precision valuest to the list, to have them for every cloud pairs
            recList.append(recMean)
            precList.append(precMean)

            # Persist prev data
            save = open(self.persistence_file, "w")
            save.write("{0}\n".format(pairnum))
            ############################
            for auc_pair in auc_pairs:
                save.write("{0}|".format(auc_pair))
            save.write("\n")
            ############################
            for auc_pair in auc_pairs_2:
                save.write("{0}|".format(auc_pair))
            save.write("\n")
            ############################
            for rec in recList:
                save.write("{0}|".format(rec))
            save.write("\n")
            ############################
            for prec in precList:
                save.write("{0}|".format(prec))
            save.write("\n")
            save.close()

        f.close()

        # Aggregate the PRC-s of all cloud pairs
        recMeanAll = []
        precMeanAll = []
        for n in np.arange(0, len(recList[0])):
            recMeanAll.append(np.mean([row[n] for row in recList]))
            precMeanAll.append(np.mean([row[n] for row in precList]))

        return auc_pairs, auc_pairs_2, np.array(recMeanAll), np.array(precMeanAll)
