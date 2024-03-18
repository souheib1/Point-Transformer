"""
 Some functions / setups were inspired from those impelmentations :

    - https://github.com/yzheng97/Point-Transformer-Cls/blob/main/  
    - https://github.com/Pointcept/Pointcept
    - https://github.com/POSTECH-CVLab/point-transformer
    - https://colab.research.google.com/drive/1JqLwVHDH3N6zjSbFfWyUF7WmzqPxzAkY?usp=sharing
    
- Query/Key and Value vectors are supposed to have the same dimention as the embedding (as in the original paper)
- Each point attend to its local neighborhood (knn)
"""
import torch 
import torch.nn as nn
import numpy as np


def points_from_idx(points, idx):
    """
    gather point coordianates from a set using indices.
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class PointTransformerLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k=16):
        """
        Single layer of a Point Transformer model.
        
        Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        k (int, optional): Number of nearest neighbors to consider. Defaults to 16.
        """  
        super().__init__()
        self.k = k # number of neighbors
        self.linear1 = nn.Linear(dim_in, dim_out) # (d->f)
        self.linear2 = nn.Linear(dim_out, dim_in) # (f->d)
        self.keys = nn.Linear(dim_out, dim_out, bias=False) # psi (f->f)
        self.queries = nn.Linear(dim_out, dim_out, bias=False) #phi (f->f)
        self.values= nn.Linear(dim_out, dim_out, bias=False) # alpha (f->f)
        self.mapping = nn.Sequential(   #gamma (f->f)
                                    nn.Linear(dim_out, dim_out), 
                                    nn.ReLU(), 
                                    nn.Linear(dim_out, dim_out)
                                    )
        self.positional_encoding = nn.Sequential(  # delta (3->f)
                                    nn.Linear(3, dim_out), 
                                    nn.ReLU(), 
                                    nn.Linear(dim_out, dim_out)
                                    )

    def forward(self, coordinates, features):
        
        # Local Attetion : coordinates of nearest neighbors
        dists = torch.sum((coordinates[:, :, None] - coordinates[:, None]) ** 2, dim=-1)  # Compute pairwise squared euclidean distance 
        knn_idx = dists.argsort()[:, :, :self.k]  # Get indices of nearest neighbors 
        knn_coords = points_from_idx(coordinates, knn_idx)  # Gather coordinates of nearest neighbors 
        #print(knn_coords.shape)
        
        before = features
        x = self.linear1(features) 
        
        # Compute queries, keys, and values
        q = self.queries(x)  
        k = points_from_idx(self.keys(x), knn_idx)  
        v = points_from_idx(self.values(x), knn_idx) 
        # Compute positional encoding
        pos_enc = self.positional_encoding(coordinates[:, :, None] - knn_coords)  

        # Compute attention
        attention = self.mapping(q[:, :, None] - k + pos_enc) # compute raw attetion
        attention = torch.nn.functional.softmax(attention / np.sqrt(k.size(-1)), dim=-2) # division leads to more stable gradients.
        #print(attention.shape)
        out = torch.einsum('bmnf,bmnf->bmf', attention, v + pos_enc)  # apply attention to values 

        out = self.linear2(out)  # final linear transformation
        out += before # add residual connection 
        return out, attention



if __name__ == '__main__':
    
    print("Point Transformer Layer")
    layer = PointTransformerLayer(dim_in=10, dim_out=3, k=16)
    batch_size = 16
    npoints = 1024 
    dim_in = 10
    coords = torch.rand((batch_size, npoints, 3))
    features = torch.rand((batch_size, npoints, dim_in))
    output, attention = layer(coords, features)
    print("\t batch_size=",batch_size)
    print("\t dim_in=",dim_in)
    print("\t N_points=",npoints)
    print("\t Local_Neighborhood k=",16)
    print('New feature shape:', output.shape)

