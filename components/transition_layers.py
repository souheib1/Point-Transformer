"""
 Some functions / setups were inspired from those impelmentations :

    - https://github.com/yzheng97/Point-Transformer-Cls/blob/main/
    - https://github.com/Pointcept/Pointcept
    - https://github.com/pierrefdz/point-transformer/blob/main/point_transformer_block.py
    - https://github.com/qq456cvb/Point-Transformers/blob/master/
    - https://colab.research.google.com/drive/1JqLwVHDH3N6zjSbFfWyUF7WmzqPxzAkY#scrollTo=CkCu5dj12Upg
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def farthest_point_sample(points, npoint):
    """
    Identify a well-spread subset P2 ⊂ P1 with the requisite cardinality
    Input:
        points: pointcloud data
        npoint: number of samples
    """
    device = points.device
    batch_size, N, _ = points.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, N).to(device) * 1e10
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.to(device)


def points_from_idx(points, idx):
    """
    gather point coordianates from a set using indices.
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class TransitionDownLayer(nn.Module):
    def __init__(self, npoint, k, input_dim, output_dim):
        """
        Transition Down Layer 
        Input :
            npoint: target number of points after transition down
            k: number of neighbors to max pool the new features from
            input_dim: dimension of input features for each point
            output_dim: dimension of output features for each point
        """
        super().__init__()
        self.npoint = npoint
        self.k = k
        
        # MLP layers for processing point features 
        self.mlp_convs = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [batch_size, N, 3]
            features: input points data, [batch_size, N, in_dim]
        """
        # Farthest point sampling
        fps_idx = farthest_point_sample(xyz, self.npoint) 
        new_xyz = points_from_idx(xyz, fps_idx) 
        torch.cuda.empty_cache()
        # Find k nearest neighbors
        dists = torch.sum(((new_xyz[:, :, None] - xyz[:, None]) ** 2), dim=-1)
        idx = dists.argsort()[:, :, :self.k]  
        torch.cuda.empty_cache()
        
        # Extract features for each point and perform max pooling
        new_features = features.transpose(1,2) 
        new_features = self.mlp_convs(new_features) 
        new_features = new_features.transpose(1,2) 
        grouped_features = points_from_idx(new_features, idx) 
        new_features, _ = torch.max(grouped_features, 2) 
        return new_xyz, new_features


       
# class TransitionUpBlock(nn.Module): # we don't need it for the classifier 
#     def __init__(self, channels_in, channels_out):
#         super().__init__()
#         pass

#     def forward(self, input):
#         pass
    
    
    
if __name__=="__main__":
    # test the transition down layer
    batch_size = 16
    nb_points = 1024
    dim_in = 10
    layer = TransitionDownLayer(npoint=nb_points//2, k=16, input_dim=dim_in, output_dim=dim_in*2)
    coords = torch.rand((batch_size, nb_points, 3))
    features = torch.rand((batch_size, nb_points, dim_in))
    print("\t batch_size=",batch_size)
    print("\t dim_in=",dim_in)
    print("\t N_points=",nb_points)
    print('Input features shape: ', features.shape)
    print('Output coordinates shape (batch size, number of points, coordinates dimension) : ', 
            layer.forward(coords, features)[0].shape)
    print('Output features shape (batch size, number of points, features dimension) : ',
          layer.forward(coords, features)[1].shape)    
    