import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

"""
    Multi-Head Attention Layer using PyTorch Geometric
"""

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim, out_dim_per_head, num_heads, use_bias):
        super().__init__(aggr='add')  # Use 'add' aggregation
        self.out_dim_per_head = out_dim_per_head
        self.num_heads = num_heads
        self.use_bias = use_bias

        # Linear projections for query, key, and value
        self.Q = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.sqrt_d = out_dim_per_head ** 0.5

    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        Q_h = self.Q(x)  # [num_nodes, num_heads * out_dim_per_head]
        K_h = self.K(x)
        V_h = self.V(x)

        # Reshape into [num_nodes, num_heads, out_dim_per_head]
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim_per_head)
        K_h = K_h.view(-1, self.num_heads, self.out_dim_per_head)
        V_h = V_h.view(-1, self.num_heads, self.out_dim_per_head)

        # Perform message passing
        out = self.propagate(edge_index, Q_h=Q_h, K_h=K_h, V_h=V_h)
        return out  # [num_nodes, num_heads, out_dim_per_head]

    def message(self, Q_h_i, K_h_j, V_h_j, index, ptr, size_i):
        # Compute attention scores
        # Q_h_i: [num_edges, num_heads, out_dim_per_head]
        # K_h_j: [num_edges, num_heads, out_dim_per_head]
        score = torch.sum(Q_h_i * K_h_j, dim=-1) / self.sqrt_d  # [num_edges, num_heads]
        score = torch.clamp(score, -5, 5)  # For numerical stability

        # Flatten score and index for softmax
        num_edges = score.size(0)
        score = score.view(-1)  # [num_edges * num_heads]
        index = index.unsqueeze(1).repeat(1, self.num_heads).view(-1)  # [num_edges * num_heads]

        # Debug: Check index bounds
        max_idx = index.max().item()
        num_nodes = size_i
        print(f"Max index in MultiHeadAttentionLayer.message: {max_idx}, num_nodes: {num_nodes}")
        if max_idx >= num_nodes:
            raise ValueError(f"Invalid index detected! index={max_idx} >= num_nodes={num_nodes}")

        # Compute softmax over attention scores
        alpha = softmax(score, index)  # [num_edges * num_heads]
        alpha = alpha.view(num_edges, self.num_heads, 1)  # [num_edges, num_heads, 1]

        # Return messages: V_h_j * alpha
        return V_h_j * alpha  # [num_edges, num_heads, out_dim_per_head]

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Aggregate messages
        # inputs: [num_edges, num_heads, out_dim_per_head]
        return scatter_add(inputs, index, dim=0, dim_size=dim_size)

    def update(self, aggr_out):
        # aggr_out: [num_nodes, num_heads, out_dim_per_head]
        # Return updated node features
        return aggr_out

"""
    Graph Transformer Layer using PyTorch Geometric
"""

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm

        out_dim_per_head = out_dim // num_heads

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim_per_head, num_heads, use_bias)
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # Feedforward Network
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
         # **Add a projection layer for residual connections if dimensions mismatch**
        if self.residual and self.in_dim != self.out_dim:
            self.residual_proj = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index):
        h_in1 = x  # For residual connection
        num_nodes = x.size(0)
        print(f"GraphTransformerLayer input x shape: {x.shape}, num_nodes: {num_nodes}")
        # Multi-head attention
        attn_out = self.attention(x, edge_index)  # [num_nodes, num_heads, out_dim_per_head]
        print(f"Attention output shape: {attn_out.shape}")
        h = attn_out.view(-1, self.out_dim)  # [num_nodes, out_dim]
        print(f"After view, h shape: {h.shape}")

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        print(f"After linear O, h shape: {h.shape}")

        # if self.residual:
        #     h = h_in1 + h  # Residual connection
        #     print(f"After residual connection, h shape: {h.shape}")

        if self.residual:
            if self.residual_proj:
                h_in1 = self.residual_proj(h_in1)
                print(f"After residual projection, h_in1 shape: {h_in1.shape}")
            h = h_in1 + h  # Residual connection
            print(f"After residual connection, h shape: {h.shape}")

        if self.layer_norm:
            h = self.layer_norm1(h)
            print(f"After layer_norm1, h shape: {h.shape}")
            
        if self.batch_norm:
            h = self.batch_norm1(h)
            print(f"After batch_norm1, h shape: {h.shape}")
        
        h_in2 = h  # For second residual connection

        # Feedforward Network
        h = self.FFN_layer1(h)
        print(f"After FFN_layer1, h shape: {h.shape}")
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
        print(f"After FFN_layer2, h shape: {h.shape}")

        if self.residual:
            h = h_in2 + h  # Residual connection
            print(f"After second residual connection, h shape: {h.shape}")
            
        if self.layer_norm:
            h = self.layer_norm2(h)
            print(f"After layer_norm2, h shape: {h.shape}")
            
        if self.batch_norm:
            h = self.batch_norm2(h)   
            print(f"After batch_norm2, h shape: {h.shape}")
        return h
