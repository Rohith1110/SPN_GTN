import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim_per_head, num_heads, use_bias):
        super(MultiHeadAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim_per_head = out_dim_per_head
        self.num_heads = num_heads
        self.use_bias = use_bias

        # Linear projections for query, key, value, and edge features
        self.Q = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(edge_dim, out_dim_per_head * num_heads, bias=use_bias)
        self.sqrt_d = out_dim_per_head ** 0.5

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_heads = self.num_heads
        out_dim_per_head = self.out_dim_per_head

        # Compute Q, K, V, proj_e
        Q = self.Q(x).view(num_nodes, num_heads, out_dim_per_head)  # [num_nodes, num_heads, out_dim_per_head]
        K = self.K(x).view(num_nodes, num_heads, out_dim_per_head)
        V = self.V(x).view(num_nodes, num_heads, out_dim_per_head)
        proj_e = self.proj_e(edge_attr).view(num_edges, num_heads, out_dim_per_head)  # [num_edges, num_heads, out_dim_per_head]

        # For each edge from node j to node i
        edge_src = edge_index[0]  # source node indices j
        edge_dst = edge_index[1]  # target node indices i

        K_j = K[edge_src]  # [num_edges, num_heads, out_dim_per_head]
        Q_i = Q[edge_dst]  # [num_edges, num_heads, out_dim_per_head]
        V_j = V[edge_src]  # [num_edges, num_heads, out_dim_per_head]

        # Compute attention scores
        score = (Q_i * K_j) / self.sqrt_d  # Element-wise multiplication and scaling
        score = score * proj_e  # Incorporate edge features
        score = score.sum(dim=-1)  # Sum over feature dimensions to get [num_edges, num_heads]

        # Save pre-softmax attention scores for edges
        e_out = score.clone()  # [num_edges, num_heads]

        # Apply exponential for numerical stability
        score = torch.exp(score.clamp(-5, 5))  # [num_edges, num_heads]

        # Compute normalization factor z_i for each node i and head h
        z = scatter_add(score, edge_dst, dim=0, dim_size=num_nodes)  # [num_nodes, num_heads]

        # Compute attention coefficients
        alpha = score / (z[edge_dst] + 1e-6)  # [num_edges, num_heads]

        # Apply attention weights to V_j
        alpha = alpha.unsqueeze(-1)  # [num_edges, num_heads, 1]
        message = V_j * alpha  # [num_edges, num_heads, out_dim_per_head]

        # Aggregate messages at destination nodes
        out = scatter_add(message, edge_dst, dim=0, dim_size=num_nodes)  # [num_nodes, num_heads, out_dim_per_head]

        # Reshape output
        out = out.view(num_nodes, num_heads * out_dim_per_head)  # [num_nodes, out_dim]

        # For e_out, you can project it to match the edge feature dimension if necessary
        return out, e_out  # [num_nodes, out_dim], [num_edges, num_heads]


class GraphTransformerEdgeLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, num_heads, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super(GraphTransformerEdgeLayer, self).__init__()

        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads."
        out_dim_per_head = out_dim // num_heads

        self.attention = MultiHeadAttentionLayer(in_dim, edge_dim, out_dim_per_head, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        # Project e_attn_out to edge_dim
        self.edge_project = nn.Linear(num_heads, edge_dim)
        self.O_e = nn.Linear(edge_dim, edge_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(edge_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(edge_dim)

        # Feedforward Network for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # Feedforward Network for e
        self.FFN_e_layer1 = nn.Linear(edge_dim, edge_dim * 2)
        self.FFN_e_layer2 = nn.Linear(edge_dim * 2, edge_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(edge_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(edge_dim)

        # Add projection layers for residual connections if dimensions mismatch
        if self.residual and self.in_dim != self.out_dim:
            self.residual_proj_h = nn.Linear(self.in_dim, self.out_dim)
        else:
            self.residual_proj_h = None

        if self.residual and self.edge_dim != edge_dim:
            self.residual_proj_e = nn.Linear(self.edge_dim, edge_dim)
        else:
            self.residual_proj_e = None

    def forward(self, x, edge_index, edge_attr):
        h_in1 = x  # [num_nodes, in_dim]
        e_in1 = edge_attr  # [num_edges, edge_dim]

        # Multi-head attention
        h_attn_out, e_attn_out = self.attention(x, edge_index, edge_attr)
        # h_attn_out: [num_nodes, out_dim]
        # e_attn_out: [num_edges, num_heads]

        # Project e_attn_out to edge_dim
        e_attn_out = self.edge_project(e_attn_out)  # [num_edges, edge_dim]

        # Apply dropout
        h = F.dropout(h_attn_out, self.dropout, training=self.training)
        e = F.dropout(e_attn_out, self.dropout, training=self.training)

        # Linear transformations
        h = self.O_h(h)
        e = self.O_e(e)

        # Residual connections
        if self.residual:
            if self.residual_proj_h:
                h_in1 = self.residual_proj_h(h_in1)
            h = h_in1 + h

            if self.residual_proj_e:
                e_in1 = self.residual_proj_e(e_in1)
            e = e_in1 + e

        # Layer normalization
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        # Batch normalization
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h
        e_in2 = e

        # Feedforward network for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # Feedforward network for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        # Second residual connections
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        # Second layer normalization
        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        # Second batch normalization
        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e  # [num_nodes, out_dim], [num_edges, edge_dim]
