# Components

## Data loaders : 
Two data loader modules, data_loader.py and data_loader2.py are  designed to handle the loading and preprocessing of point cloud data for Transformer. Here I used the ModelNet40 Dataset (of 40 categories) with 9843 shapes for the training and 2468 for the testing. The data_loader.py was ispired from the TP6 and defines a custom dataset class named PointCloudData_RAM, tailored for loading point cloud  data stored in PLY format. This class utilizes transformations such as random rotation and noise addition to augment the data, along with PyTorch data loaders for efficient batch processing. Conversely, the data_loader2.py module was taken from the [PointNet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py
) implemtation. It includes functionalities for raw data processing, such as point cloud normalization and subsampling using FPS, alongside PyTorch data loaders for training and testing data.

PS : The data_loader.py generates batchs of points of the shape (batch_size,1024,3) while the data_loader2.py leverages the normals as additional channels and loads batchs of points of the shape (batch_size,1024,6).

## Samplers :
The samplers.py module contains implementations of two sampling techniques: Farthest Point Sampling (FPS) and density-based sampling. FPS sampling selects a subset of points by iteratively choosing the farthest point from the already selected ones. This technique is valuable for downsampling point clouds while preserving their spatial distribution.

On the other hand, density-based sampling selects points based on their local density, with points having higher densities being more likely to be selected. This approach enables focusing on regions with high information density in the point cloud, facilitating more effective feature extraction and representation.


## Point Transformer Layer: 
The module implements a single Point Transformer layer. Getting inspiration of various available implementations on Github such as [1](https://github.com/Pointcept/Pointcept) [2](https://github.com/POSTECH-CVLab/point-transformer) and [3](https://colab.research.google.com/drive/1JqLwVHDH3N6zjSbFfWyUF7WmzqPxzAkY?usp=sharing), I implemented the PointTransformerLayer to perform operations on the point cloud such as local attention, feature transformation, and positional encoding to process the input features and coordinates of point cloud data. The PointTransformerLayer class takes input feature dimensions (dim_in) and output feature dimensions (dim_out) as parameters, with an optional parameter k to specify the number of nearest neighbors considered for local attention. During forward propagation, the layer computes local attention based on the pairwise Euclidean distance between points, followed by queries, keys, and values computations. It then applies positional encoding and attention mechanisms to generate the final output features. Additionally, we used residual connections to preserve information from the input features.

## Transition Down Layer: 
Transition Down Layer  is a crucial component in point cloud neural network architectures. The implementation is inspired by various existing implementations and follows the design principles found in the original paper. The main objective of the layer is to reduce the number of points in the input point cloud while preserving important features. It achieves this by performing farthest point sampling to select a subset of points and then finding the k nearest neighbors for each selected point. It then aggregates features from these neighbors using max pooling, followed by convolutional operations.
