"""
Original code from https://github.com/kenziyuliu/ms-g3d

Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020).
Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 143–152).
"""

import torch
import torch.nn as nn
import numpy as np

from models.msg3d.mlp import MLP
from models.msg3d.activation import activation_factory
from util.graph import get_k_adjacency
from graph.tools import k_adjacency, normalize_adjacency_matrix

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [get_k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows

        # Build graphs
        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        # Perform Graph Convolution
        res = self.residual(x)
        agg = torch.einsum('vu,nctu->nctv', A, x)
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)

