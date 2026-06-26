"""
Graph Ecosystem Masterclass - Phase 5: Graph Neural Networks (GNNs) & Generative Models
========================================================================================
This module implements modern Graph Deep Learning. It contains a custom GNN Message Passing
framework, GCN, GraphSAGE, GAT, Graph Transformers, Relational GCN, Temporal GNNs,
Deep Graph Infomax (DGI), and VGAE—all built from scratch in raw PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional
from src.phase1_foundations import Graph

# ==========================================
# 1. Custom Message Passing Framework
# ==========================================

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Utility function in native PyTorch to perform scatter-add aggregation.
    Accumulates values from src into out at the indices specified in index.
    Acts as a lightweight, zero-dependency alternative to torch_scatter.
    Generalised to support arbitrary dimensions (e.g. 2D, 3D, etc.).
    
    Args:
        src (Tensor): Source features of shape (num_edges, ...)
        index (Tensor): Target indices of shape (num_edges,)
        dim_size (int): Size of the output dimension (usually num_nodes)
    """
    # Create empty output tensor matching src shape except at dimension 0
    out_shape = list(src.shape)
    out_shape[0] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    # Reshape index to have same number of dimensions as src
    # e.g., if src is (num_edges, d1, d2), index becomes (num_edges, 1, 1)
    index_shape = [index.size(0)] + [1] * (src.dim() - 1)
    index_expanded = index.view(index_shape).expand_as(src)
    
    # Perform scatter add along dimension 0
    out.scatter_add_(0, index_expanded, src)
    return out

class MessagePassing(nn.Module):
    """
    Base class for writing custom GNN layers from scratch.
    Emulates the message, aggregate, update design pattern.
    
    Pipeline:
      1. propagate(edge_index, x)
      2. message(x_i, x_j) (where j is source/neighbor, i is target/central node)
      3. aggregate(messages, index) (sum, mean, or max)
      4. update(aggr_out, x)
    """
    def __init__(self, aggr: str = "sum"):
        super().__init__()
        self.aggr = aggr
        assert aggr in ["sum", "mean", "max"]
        
    def propagate(self, edge_index: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Propagates messages across edges.
        
        Args:
            edge_index (Tensor): Graph connectivity of shape (2, num_edges)
                                 where edge_index[0] is source (j) and edge_index[1] is target (i).
            x (Tensor): Node features of shape (num_nodes, dim)
        """
        num_nodes = x.size(0)
        row, col = edge_index[0], edge_index[1] # row is source (j), col is target (i)
        
        # 1. Get features of source and target nodes for every edge
        x_j = x[row]  # Source features: (num_edges, dim)
        x_i = x[col]  # Target features: (num_edges, dim)
        
        # 2. Compute messages
        # Subclasses can override message() to perform custom computations
        messages = self.message(x_i, x_j, **kwargs) # (num_edges, msg_dim)
        
        # 3. Aggregate messages at target nodes
        if self.aggr == "sum":
            aggr_out = scatter_add(messages, col, num_nodes)
        elif self.aggr == "mean":
            # Sum and divide by node degrees
            sum_out = scatter_add(messages, col, num_nodes)
            # Compute degree (number of incoming edges for each target node)
            ones = torch.ones(messages.size(0), 1, dtype=x.dtype, device=x.device)
            deg = scatter_add(ones, col, num_nodes)
            # Safe division
            aggr_out = sum_out / torch.clamp(deg, min=1.0)
        elif self.aggr == "max":
            # For scatter-max, initialize with large negative number
            out = torch.full((num_nodes, messages.size(-1)), -1e15, dtype=x.dtype, device=x.device)
            index_expanded = col.unsqueeze(-1).expand_as(messages)
            aggr_out = torch.scatter_reduce(out, 0, index_expanded, messages, reduce="amax", include_self=False)
            # Replace placeholder values with 0
            aggr_out = torch.clamp(aggr_out, min=0.0)
            
        # 4. Update node features
        out = self.update(aggr_out, x)
        return out
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, **kwargs) -> torch.Tensor:
        """Constructs messages from neighbor node j to target node i."""
        return x_j
        
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Updates node features using the aggregated messages."""
        return aggr_out


# ==========================================
# 2. Canonical GNN Layers (GCN, SAGE, GAT)
# ==========================================

class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) layer.
    Mathematical formulation:
        h_i = \sigma( W * \sum_{j \in N(i) \cup {i}} \frac{1}{\sqrt{d_i d_j}} h_j )
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="sum")
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 1. Add self-loops to edge_index: A_tilde = A + I
        num_nodes = x.size(0)
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        # 2. Pre-multiply node features by weight matrix
        # (Writing linear transform before message passing is highly optimized)
        x = self.linear(x)
        
        # 3. Compute symmetric normalization factors: 1 / sqrt(d_i * d_j)
        row, col = edge_index[0], edge_index[1]
        
        # Compute degrees (out-degree in undirected setting = total degree)
        ones = torch.ones(edge_index.size(1), 1, dtype=x.dtype, device=x.device)
        deg = scatter_add(ones, col, num_nodes).squeeze(-1) # (num_nodes,)
        
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Edge weights: norm = 1 / sqrt(d_row * d_col)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # (num_edges,)
        norm = norm.unsqueeze(-1) # (num_edges, 1)
        
        # 4. Propagate messages
        out = self.propagate(edge_index, x, norm=norm)
        return out + self.bias
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Scale neighbor features by symmetric normalization factor
        return norm * x_j

class GraphSAGEConv(MessagePassing):
    """
    GraphSAGE Layer (Sample and Aggregate).
    Formulation:
        h_{N(i)} = Aggregate( {h_j, \forall j \in N(i)} )
        h_i = \sigma( W * concat(h_i, h_{N(i)}) )
    """
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 1. Propagate neighbor messages
        # No self-loops here; neighborhood is aggregated separately from the node itself
        aggr_neigh = self.propagate(edge_index, x)
        
        # 2. Combine central node and neighbor features
        out = self.lin_self(x) + self.lin_neigh(aggr_neigh)
        return F.relu(out + self.bias)

class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) Layer.
    Computes multi-head self-attention over neighborhoods.
    Formulation:
        alpha_{i,j} = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )
        h_i = \sum_{j \in N(i)} \alpha_{i,j} W h_j
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True):
        super().__init__(aggr="sum")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        
        # Node projection weight
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameter vectors for source and target projection
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        
        # Initialize
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        
        # 1. Linear projection: (num_nodes, heads * out_channels)
        h = self.lin(x)
        # Reshape to separate heads: (num_nodes, heads, out_channels)
        h = h.view(-1, self.heads, self.out_channels)
        
        # 2. Calculate attention coefficients pre-aggregating
        # Dot product with attention parameter vectors
        alpha_src = (h * self.att_src).sum(dim=-1) # (num_nodes, heads)
        alpha_dst = (h * self.att_dst).sum(dim=-1) # (num_nodes, heads)
        
        row, col = edge_index[0], edge_index[1]
        
        # Edge attention scores: e_ij = LeakyReLU(alpha_src[j] + alpha_dst[i])
        edge_attn = alpha_src[row] + alpha_dst[col] # (num_edges, heads)
        edge_attn = F.leaky_relu(edge_attn, negative_slope=0.2)
        
        # Softmax normalization over target node col (i)
        # To do softmax safely, we subtract the max value per target node
        attn_max = scatter_add(edge_attn, col, num_nodes) # (num_nodes, heads)
        # Map back to edges
        edge_attn_max = attn_max[col]
        # Exponential
        edge_exp = torch.exp(edge_attn - edge_attn_max)
        # Denominator sum
        edge_denom = scatter_add(edge_exp, col, num_nodes)
        # Normalised attention coefficients: (num_edges, heads)
        alpha = edge_exp / (edge_denom[col] + 1e-15)
        
        # 3. Propagate messages scaled by attention
        out = self.propagate(edge_index, x=h, alpha=alpha) # (num_nodes, heads, out_channels)
        
        # 4. Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        return out + self.bias
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # x_j: (num_edges, heads, out_channels)
        # alpha: (num_edges, heads)
        # Multiply along heads dimensions
        return alpha.unsqueeze(-1) * x_j


# ==========================================
# 3. Graph Transformer
# ==========================================

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer.
    Computes global-style Multi-Head Attention over neighborhood edges.
    Can incorporate Laplacian Positional Encodings (LPEs) to inject structural awareness.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.d_k = out_channels // heads
        
        self.q_lin = nn.Linear(in_channels, out_channels)
        self.k_lin = nn.Linear(in_channels, out_channels)
        self.v_lin = nn.Linear(in_channels, out_channels)
        
        # Projection shortcut for residual connection if dimensions differ
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        self.out_lin = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        
        # 1. Compute Q, K, V projections
        q = self.q_lin(x).view(-1, self.heads, self.d_k) # (N, heads, d_k)
        k = self.k_lin(x).view(-1, self.heads, self.d_k) # (N, heads, d_k)
        v = self.v_lin(x).view(-1, self.heads, self.d_k) # (N, heads, d_k)
        
        row, col = edge_index[0], edge_index[1] # row is source (j), col is target (i)
        
        # 2. Compute scaled dot-product attention coefficients on edges
        # score_ij = (Q_i^T * K_j) / sqrt(d_k)
        q_i = q[col] # Target queries: (num_edges, heads, d_k)
        k_j = k[row] # Source keys: (num_edges, heads, d_k)
        
        attn_scores = (q_i * k_j).sum(dim=-1) / np.sqrt(self.d_k) # (num_edges, heads)
        
        # Softmax over neighborhood
        scores_max = scatter_add(attn_scores, col, num_nodes)
        edge_exp = torch.exp(attn_scores - scores_max[col])
        edge_denom = scatter_add(edge_exp, col, num_nodes)
        alpha = edge_exp / (edge_denom[col] + 1e-15) # (num_edges, heads)
        
        # 3. Aggregate values scaled by attention
        v_j = v[row] # (num_edges, heads, d_k)
        weighted_val = alpha.unsqueeze(-1) * v_j # (num_edges, heads, d_k)
        aggr_out = scatter_add(weighted_val, col, num_nodes) # (N, heads, d_k)
        
        # Flatten heads
        h_attn = aggr_out.view(-1, self.out_channels)
        
        # 4. Residual and LayerNorm
        h_attn = self.norm1(self.proj(x) + self.out_lin(h_attn))
        
        # 5. Feed Forward Network + Residual
        out = self.norm2(h_attn + self.ffn(h_attn))
        return out


# ==========================================
# 4. Advanced GNNs (RGCN, Temporal, DGI)
# ==========================================

class RGCNConv(nn.Module):
    """
    Relational Graph Convolution Network (RGCN) Layer for heterogeneous graphs.
    Applies relation-specific weight matrices during aggregation.
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        # Relation-specific weights
        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        # Self-loop weight
        self.weight_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index: shape (2, num_edges)
            edge_type: shape (num_edges,) containing relation IDs [0, num_relations-1]
        """
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_channels, dtype=x.dtype, device=x.device)
        
        row, col = edge_index[0], edge_index[1]
        
        # Aggregate for each relation type independently
        for r in range(self.num_relations):
            # Mask edges of relation type r
            mask = (edge_type == r)
            if not mask.any():
                continue
                
            r_row, r_col = row[mask], col[mask]
            
            # Map source features using relation weight matrix W_r
            # x: (N, in_channels) -> W_r: (in_channels, out_channels) -> (N, out_channels)
            x_transformed = torch.matmul(x, self.weights[r])
            
            # Message from neighbor: scale by relation-specific degree normalization
            ones = torch.ones(r_col.size(0), 1, dtype=x.dtype, device=x.device)
            deg = scatter_add(ones, r_col, num_nodes).squeeze(-1)
            norm = 1.0 / torch.clamp(deg, min=1.0)
            
            # Aggregate neighbors
            msg = x_transformed[r_row] * norm[r_col].unsqueeze(-1)
            aggr = scatter_add(msg, r_col, num_nodes)
            out += aggr
            
        # Add self-loop transform and bias
        return out + self.weight_self(x) + self.bias

class TemporalMemoryGNN(nn.Module):
    """
    A simplified Temporal Graph Network (TGN) cell.
    Updates dynamic node states (memories) upon edge arrival events using a GRU gate.
    """
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(node_dim + edge_dim, node_dim)
        
    def forward(self, memory: torch.Tensor, u: int, v: int, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Updates memory state of nodes u and v involved in a temporal interaction event.
        """
        updated_memory = memory.clone()
        
        # Construct interaction message: concat(source_memory, target_memory, edge_feat)
        # for node u
        msg_u = torch.cat([memory[u], memory[v], edge_feat], dim=-1).unsqueeze(0) # (1, dim)
        updated_memory[u] = self.gru(msg_u, memory[u].unsqueeze(0)).squeeze(0)
        
        # for node v (symmetry)
        msg_v = torch.cat([memory[v], memory[u], edge_feat], dim=-1).unsqueeze(0)
        updated_memory[v] = self.gru(msg_v, memory[v].unsqueeze(0)).squeeze(0)
        
        return updated_memory

class DeepGraphInfomax(nn.Module):
    """
    Deep Graph Infomax (DGI) model.
    Trains GNNs self-supervised by maximizing mutual information between local node patches
    and the global graph summary vector.
    """
    def __init__(self, encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        # Bilinear discriminator to score local-global pairs
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor, x_corrupt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): Real node features.
            x_corrupt (Tensor): Corrupted node features (e.g. row-wise shuffled).
            
        Returns:
            positive_scores: scores of true patches.
            negative_scores: scores of corrupted patches.
        """
        # 1. Encode positive view to get local node embeddings
        h_pos = self.encoder(edge_index, x) # (N, hidden_dim)
        
        # 2. Encode negative view to get corrupted embeddings
        h_neg = self.encoder(edge_index, x_corrupt) # (N, hidden_dim)
        
        # 3. Readout: global summary vector (simple average sigmoid)
        summary = torch.sigmoid(h_pos.mean(dim=0, keepdim=True)) # (1, hidden_dim)
        summary_expanded = summary.expand_as(h_pos)
        
        # 4. Discriminator scores
        pos_scores = self.discriminator(h_pos, summary_expanded) # (N, 1)
        neg_scores = self.discriminator(h_neg, summary_expanded) # (N, 1)
        
        return pos_scores, neg_scores


# ==========================================
# 5. Generative Models (GAE & VGAE)
# ==========================================

class VGAE(nn.Module):
    """
    Variational Graph Autoencoder (VGAE).
    Encodes graph nodes into a distribution (mean, logvar) and decodes by reconstructing
    the adjacency matrix via inner product.
    """
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        # Shared GCN Layer
        self.gcn_shared = GCNConv(in_channels, 2 * latent_dim)
        # Latent Mean layer
        self.gcn_mu = GCNConv(2 * latent_dim, latent_dim)
        # Latent Log-Variance layer
        self.gcn_logvar = GCNConv(2 * latent_dim, latent_dim)
        
    def encode(self, edge_index: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.gcn_shared(edge_index, x))
        mu = self.gcn_mu(edge_index, h)
        logvar = self.gcn_logvar(edge_index, h)
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes by computing the reconstructed probability adjacency matrix: sigmoid(Z * Z^T)."""
        adj_rec = torch.sigmoid(torch.matmul(z, z.T))
        return adj_rec

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Computes KL Divergence loss against a standard normal prior."""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(edge_index, x)
        z = self.reparameterize(mu, logvar)
        adj_rec = self.decode(z)
        return adj_rec, mu, logvar

class GraphDiffusionSimulator:
    """Simulates physical heat or information diffusion processes on graph topologies."""
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def run_diffusion(self, initial_heat: np.ndarray, time_steps: int = 5, alpha: float = 0.2) -> List[np.ndarray]:
        """
        Simulates discrete-time graph heat diffusion:
        h_t+1 = h_t - alpha * L * h_t = (I - alpha * L) * h_t
        where L is the combinatorial Graph Laplacian.
        """
        L, _ = self.graph.laplacian_matrix()
        n = L.shape[0]
        I = np.eye(n)
        
        # Diffusion propagation matrix
        W = I - alpha * L
        
        heat_history = [initial_heat.copy()]
        curr_heat = initial_heat.copy()
        
        for _ in range(time_steps):
            curr_heat = np.dot(W, curr_heat)
            heat_history.append(curr_heat.copy())
            
        return heat_history
