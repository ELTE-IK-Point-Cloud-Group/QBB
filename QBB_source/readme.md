# QBB - Quantile-Based Binarization of 3D Point Cloud Descriptors
## `QBB Source`

In this folder you can find our new binarization method's detailed codes.

### Description of the files:
- `QBB_binarization.py`
  
  This is the main binarizer class. This binarizer object needs a dimension binarizer object (detailed below). Then it can binarize a feature descriptor according to the precalculated statistics, which defines the endpoints and the groups for each dimension.

- `QBB_dim_binarizer.py`

  This object can binarize a dimension of a feature descriptor with the given endpoints. The method is described in our article.

  The method can be used with different codes for the groups. Those different codes were implemented in different files.

- `QBB_dim_gray_binarizer.py`

  This is the GRAY code version of the dimension binarizer. The only difference is in the content of the `bin_array` array, which holds all the group's binary codes.

  The reason for this different binary code is described in our paper.

- `QBB_dim_mersenne_binarizer.py`

  This is the MERSENNE code version of the dimension binarizer. The only difference is in the content of the `bin_array` array, which holds all the group's binary codes.

  The reason for this different binary code is described in our paper.

- `QBB_endpoints.py`

  This small class is used during the statistics generation. Holds all the possible endpoints for all the dimensions of the dataset.

- `QBB_statistics_for_descriptor.py`

  This class is used to generate the groups endpoints for a given dataset.
  Loads all the feature descriptors in the folder and calculates the theoretical best group numbers and endpoints for those groups with the given maximal usable bit number for each dimension.
