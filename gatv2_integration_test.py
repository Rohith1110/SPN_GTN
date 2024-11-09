import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Module
from models import CRF, MLP  # Ensure CRF and MLP are defined in models.py
from typing import Namespace
from collections import OrderedDict

class GATv2NodeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.6):
        super(GATv2NodeGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GATv2EdgeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.6):
        super(GATv2EdgeGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_repr, edge_index):
        edge_repr = self.conv1(edge_repr, edge_index)
        edge_repr = self.elu(edge_repr)
        edge_repr = self.dropout(edge_repr)
        edge_repr = self.conv2(edge_repr, edge_index)
        return edge_repr

class CustomSPNModel(nn.Module):
    def __init__(self, args: Namespace, num_features: int, num_classes: int = 2):
        super(CustomSPNModel, self).__init__()
        self.args = args
        hidden_channels = args.hidden_channels  # Define hidden_channels in your args

        # Node GNN using GATv2
        self.node_gnn = GATv2NodeGNN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=args.gatv2_heads,   # Define gatv2_heads in your args
            dropout=args.dropout_prob
        )

        # Edge GNN using GATv2
        self.edge_gnn = GATv2EdgeGNN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=args.gatv2_heads,   # Define gatv2_heads in your args
            dropout=args.dropout_prob
        )

        # CRF Layer
        self.crf = CRF(args, hidden_channels, num_classes)

        # Proxy Optimization Layers (if applicable)
        self.proxy = MLP(
            in_features=hidden_channels,
            out_features=num_classes,
            hidden_sizes=args.proxy_hidden_sizes,  # Define in your args
            activation=args.proxy_activation,      # Define in your args
            norm_module=args.proxy_norm_module,    # Define in your args
            dropout_prob=args.proxy_dropout_prob    # Define in your args
        )

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # Node representation via GATv2
        node_repr = self.node_gnn(x, edge_index)

        # Edge representation via GATv2
        edge_repr = self.edge_gnn(edge_type, edge_index)

        # CRF Processing
        crf_output = self.crf(node_repr, edge_repr, data)

        # Proxy Optimization (if applicable)
        proxy_output = self.proxy(node_repr)

        return crf_output, proxy_output