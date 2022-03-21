# QBB - Quantile-Based Binarization of 3D Point Cloud Descriptors
## Descriptiveness Test

## Setup
The test requires a few parameters, which can be defined in the test_settings.json file.
- "sampleNum": Number of points chosen from the clouds for the test.
- "keypoint_support_radius": The search radius for every point, same as the feature estimation search radius.
- "repeat": Times to repeat the test. The result is the average of the repeats.
- "cloud_pairs": Number of cloud pairs to use in the test.
- "dim": Dimension of the binary descriptor. Used in custom metrics test.
- "descriptor":  Name of the folder containing the feature descriptors.
- "dataset_name": Name of the folder containing all data for the dataset. 
- "dataset_path": Where to find the dataset folder.
- "input_cloud_format": File format of the clouds. (.pcd or .ply) 
- "persistence_file": File to save results while running. Can be useful if the script needs to be stopped and continued later.
- "is_binary": The feature descriptor is binary or real valued. (Binary versions are stored within mpz numbers.)
- "read_persisted_data": Start with reading the given persistence file or start from zero.
- "use_custom_metric": For binary feature descriptors you can choose between the default Hamming distance and the weighted custom metric.
- "bits": In binary descriptors, how many bits represents a dimension of the real valued descriptor.

## File structure
```
├─ "dataset_path"
│   ├─ dataset_1
│   ├─ dataset_2
│   ├─ "dataset_name"
│   │   ├─ gt.log  // ground truth file
│   │   ├─ clouds 
│   │   │   ├─ cloud_bin_0.ply / .pcd
│   │   │   ├─ cloud_bin_1.ply
│   │   │   ├─ ...
│   │   │   ├─ cloud_bin_n.ply
│   │   ├─ feature_descriptor_1
│   │   ├─ "descriptor"
│   │   │   ├─ cloud_bin_0.csv
│   │   │   ├─ cloud_bin_1.csv
│   │   │   ├─ ...
│   │   │   ├─ cloud_bin_n.csv
│   │   ├─ feature_descriptor_2
│   ├─ dataset_3
```

## Run
0. Install dependencies
1. Prepare dataset and folder structure
2. Setup parameters in `test_settings.json`
3. Run the `main.py` script with python3

## Results
Results are generated in three folders:
- 'logs': 
    - settings
    - mean of auc values
    - standard deviation of auc values
    - average of auc values for every cloud pair
    - auc values for every repeat and every pair
- 'logs_image':
    - Precision and recall values for the repeats
- 'results':
    - auc results in a csv format