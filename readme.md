# QBB - Quantile-Based Binarization of 3D Point Cloud Descriptors

This repository was created to share codes which was created for the paper:  
[QBB - Quantile-Based Binarization of 3D Point Cloud Descriptors]()

Written by:
- Dániel Varga
- János Márk Szalai-Gindl
- Márton Ambrus-Dobai
- Sándor Laki

## Script Dependencies
The some of the python scripts in this repository depends on external packages. These are collected in the `requirements.txt` file and can be installed with `pip` package manager.
After that, all the scripts can be run with `python` version 3.

## Repository Content
- [QBB_source](QBB_source/readme.md)

  This is the main result of our research. The new binarization method with a detailed code.

- [binarization_methods](binarization_methods/readme.md)

  Python implementation of some binarization methods which were used for the test in our research.

- [descriptiveness_test](descriptiveness_test/readme.md)

  This folder contains the script which was used to calculate the descriptive power of each binarization method. The methods were compared with the results of these calculations.

- [examples](examples/readme.md)

  In this folder you can find example codes to see, how to use our methods.

- [standalone_binary_descriptors](standalone_binary_descriptors/readme.md)

  These are some descriptor generation methods with binary descriptors as their result.

