import torch 
import torch.nn as nn
import numpy as np

from components.point_transformer_layer import PointTransformerLayer
from components.transition_layers import TransitionDownLayer


class PointTransformerModel(nn.Module):
    def __init__(self, npoints=1024, nblocks=4, k=16, num_class=40, input_dim=6,pooling='max'):
        super().__init__()
        self.nblocks = nblocks
        self.pooling = pooling
        self.MLP1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.transformer_initial = PointTransformerLayer(32, 64, k)
        
        self.down_blocks = nn.ModuleList([TransitionDownLayer(npoints // 4 ** i, 
                                                              k, 
                                                              32 * 2 ** i // 2, 
                                                              32 * 2 ** i,
                                                              pooling=self.pooling) for i in range(1, nblocks + 1)])
        
        self.transformer_blocks = nn.ModuleList([PointTransformerLayer(32 * 2 ** i, 
                                                                       64, 
                                                                       k) for i in range(1, nblocks + 1)])
        
        self.MLP2 = nn.Sequential(
            nn.Linear(32*2**nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )
    
    def forward(self, x):
        xyz = x[..., :3]
        features,_ = self.transformer_initial(xyz,self.MLP1(x))

        for i in range(self.nblocks):
            xyz, features = self.down_blocks[i](xyz, features)
            features,_ = self.transformer_blocks[i](xyz, features)
        
        return self.MLP2(features.mean(1))
    
    
if __name__ == '__main__':
    model = PointTransformerModel()
    print(model)
    batch_size, N, input_dim = 16, 1024, 6
    features = torch.rand((batch_size, N, input_dim))
    output = model(features)
    print(output.shape)