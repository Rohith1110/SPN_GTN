import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, num_heads, use_bias):
        super().__init__()
        self.out_dim_per_head = out_dim_per_head
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.in_dim = in_dim

        # Linear projections for query, key, and value
        self.Q = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.sqrt_d = out_dim_per_head ** 0.5

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        num_heads = self.num_heads
        out_dim_per_head = self.out_dim_per_head

        # Compute Q, K, V matrices
        Q_h = self.Q(x).view(num_nodes, num_heads, out_dim_per_head)  # [num_nodes, num_heads, out_dim_per_head]
        K_h = self.K(x).view(num_nodes, num_heads, out_dim_per_head)
        V_h = self.V(x).view(num_nodes, num_heads, out_dim_per_head)

        # Get source and target node indices
        edge_src = edge_index[0]  # Source nodes
        edge_dst = edge_index[1]  # Target nodes

        # Get Q_i, K_j, V_j for each edge
        Q_h_i = Q_h[edge_dst]  # [num_edges, num_heads, out_dim_per_head]
        K_h_j = K_h[edge_src]  # [num_edges, num_heads, out_dim_per_head]
        V_h_j = V_h[edge_src]  # [num_edges, num_heads, out_dim_per_head]

        # Compute attention scores
        score = torch.sum(Q_h_i * K_h_j, dim=-1) / self.sqrt_d  # [num_edges, num_heads]
        score = torch.clamp(score, -5, 5)

        # Compute softmax over attention scores per head
        alpha = scatter_softmax(score, edge_dst, dim=0)  # [num_edges, num_heads]

        # Apply attention weights
        alpha = alpha.unsqueeze(-1)  # [num_edges, num_heads, 1]
        messages = V_h_j * alpha  # [num_edges, num_heads, out_dim_per_head]

        # Aggregate messages
        out = scatter_add(messages, edge_dst, dim=0, dim_size=num_nodes)  # [num_nodes, num_heads, out_dim_per_head]

        # Reshape output
        out = out.view(num_nodes, num_heads * out_dim_per_head)  # [num_nodes, num_heads * out_dim_per_head]

        return out  # [num_nodes, num_heads * out_dim_per_head]

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

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads."
        out_dim_per_head = out_dim // num_heads

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim_per_head, num_heads, use_bias)
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # Feedforward Network
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
        # Add a projection layer for residual connections if dimensions mismatch
        if self.residual and self.in_dim != self.out_dim:
            self.residual_proj = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.residual_proj = None
           
    def forward(self, x, edge_index):
        h_in1 = x  # For residual connection
        
        # Multi-head attention
        attn_out = self.attention(x, edge_index)  # [num_nodes, out_dim]

        # Linear transformation and dropout
        h = F.dropout(attn_out, self.dropout, training=self.training)  # [num_nodes, out_dim]
        h = self.O(h)  # [num_nodes, out_dim]

        # Residual Connection
        if self.residual:
            if self.residual_proj:
                h_in1 = self.residual_proj(h_in1)  # [num_nodes, out_dim]
            h = h_in1 + h  # [num_nodes, out_dim]

        # Layer Normalization
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        # Batch Normalization
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h  # For second residual connection

        # Feedforward Network
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # [num_nodes, out_dim]

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)   
        
        return h  # [num_nodes, out_dim]
