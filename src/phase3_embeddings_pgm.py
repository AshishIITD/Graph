"""
Graph Ecosystem Masterclass - Phase 3: Graph Embeddings & Probabilistic Graphical Models (PGMs)
=============================================================================================
This module bridges the gap between classical structural algorithms and statistical learning on graphs.
It implements Node2Vec, LINE, and HOPE embeddings using PyTorch/SVD, and classical PGMs
including Bayesian Networks, HMM (Viterbi), and MRF belief propagation from scratch.
"""

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Set, Union, Optional
from src.phase1_foundations import Graph

# ==========================================
# 1. Graph Embeddings (Node2Vec, LINE, HOPE)
# ==========================================

class BiasedRandomWalker:
    """
    Implements biased random walks on graphs for DeepWalk and Node2Vec.
    - DeepWalk: p = 1, q = 1 (uniform random walk)
    - Node2Vec: biased walk controlled by:
      - p: return parameter (controls local breadth-first search)
      - q: in-out parameter (controls global depth-first search exploration)
    """
    def __init__(self, graph: Graph, p: float = 1.0, q: float = 1.0):
        self.graph = graph
        self.p = p
        self.q = q
        
    def get_transition_prob(self, t: any, v: any) -> np.ndarray:
        """
        Computes the transition probabilities for the next step from node v, 
        given that the previous step was from node t.
        """
        neighbors = list(self.graph.adj[v].keys())
        if not neighbors:
            return np.array([])
            
        unnormalized_probs = []
        for x in neighbors:
            weight = self.graph.adj[v][x]["weight"]
            
            if x == t:
                # Case 1: Return to previous node t (d_tx = 0)
                alpha = 1.0 / self.p
            elif x in self.graph.adj[t]:
                # Case 2: Node x is connected to t (d_tx = 1)
                alpha = 1.0
            else:
                # Case 3: Node x is not connected to t (d_tx = 2)
                alpha = 1.0 / self.q
                
            unnormalized_probs.append(alpha * weight)
            
        probs = np.array(unnormalized_probs)
        return probs / np.sum(probs)

    def simulate_walk(self, start_node: any, walk_length: int) -> List[any]:
        """Simulates a single biased random walk starting from start_node."""
        walk = [start_node]
        
        while len(walk) < walk_length:
            curr = walk[-1]
            neighbors = list(self.graph.adj[curr].keys())
            if not neighbors:
                break
                
            if len(walk) == 1:
                # First step: uniform transition weighted by edge weights
                weights = [self.graph.adj[curr][nbr]["weight"] for nbr in neighbors]
                probs = np.array(weights) / sum(weights)
                next_node = np.random.choice(neighbors, p=probs)
            else:
                # Subsequent steps: biased by previous node
                prev = walk[-2]
                probs = self.get_transition_prob(prev, curr)
                next_node = np.random.choice(neighbors, p=probs)
                
            walk.append(next_node)
            
        return walk

    def generate_walks(self, num_walks: int, walk_length: int) -> List[List[any]]:
        """Generates multiple walks from every node in the graph."""
        walks = []
        nodes = list(self.graph.nodes.keys())
        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.simulate_walk(node, walk_length))
        return walks

class SkipGramModel(nn.Module):
    """A PyTorch implementation of the Skip-Gram model for learning node embeddings."""
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        # Target node embeddings
        self.w_embeddings = nn.Embedding(num_nodes, embedding_dim)
        # Context node embeddings
        self.v_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize weights
        self.w_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.v_embeddings.weight.data.zero_()
        
    def forward(self, target: torch.Tensor, context: torch.Tensor, neg_samples: torch.Tensor) -> torch.Tensor:
        """
        Computes the Negative Sampling Loss.
        Loss = -log(sigmoid(w_t^T * v_c)) - sum(log(sigmoid(-w_t^T * v_n)))
        """
        # Embed target (batch_size, embedding_dim)
        embed_target = self.w_embeddings(target)
        # Embed positive context (batch_size, embedding_dim)
        embed_context = self.v_embeddings(context)
        # Embed negative samples (batch_size, num_neg_samples, embedding_dim)
        embed_neg = self.v_embeddings(neg_samples)
        
        # Positive score: dot product of target and context
        # (batch_size, 1)
        pos_score = torch.sum(embed_target * embed_context, dim=1, keepdim=True)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-15)
        
        # Negative score: batch matrix multiplication of target and negative context
        # (batch_size, num_neg_samples, embedding_dim) x (batch_size, embedding_dim, 1) -> (batch_size, num_neg_samples)
        neg_score = torch.bmm(embed_neg, embed_target.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-15), dim=1, keepdim=True)
        
        return -torch.mean(pos_loss + neg_loss)

def train_node2vec(graph: Graph, embedding_dim: int = 16, walk_length: int = 10, 
                   num_walks: int = 5, p: float = 1.0, q: float = 1.0, 
                   epochs: int = 5, lr: float = 0.01) -> Tuple[np.ndarray, List[any]]:
    """
    Generates biased random walks and trains a Skip-Gram model in PyTorch.
    Returns:
        embeddings: trained embedding matrix of shape (N, embedding_dim).
        node_order: list of nodes matching the row indices.
    """
    node_order = list(graph.nodes.keys())
    node_to_idx = {node: i for i, node in enumerate(node_order)}
    n = len(node_order)
    
    # 1. Generate biased walks
    walker = BiasedRandomWalker(graph, p, q)
    walks = walker.generate_walks(num_walks, walk_length)
    
    # 2. Build training dataset (Target, Context) pairs using sliding window
    window_size = 2
    pairs = []
    for walk in walks:
        walk_idxs = [node_to_idx[node] for node in walk]
        for i, target in enumerate(walk_idxs):
            start = max(0, i - window_size)
            end = min(len(walk_idxs), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((target, walk_idxs[j]))
                    
    if not pairs:
        # Fallback if graph is empty/isolated
        return np.zeros((n, embedding_dim)), node_order
        
    # 3. Setup training loop
    model = SkipGramModel(n, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generate negative samples: flat uniform sampling for educational simplicity
    def get_negative_samples(batch_size, num_neg=5):
        return torch.randint(0, n, (batch_size, num_neg))
        
    targets_t = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts_t = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    
    dataset_size = len(pairs)
    batch_size = 64
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_t = targets_t[indices]
            batch_c = contexts_t[indices]
            batch_neg = get_negative_samples(len(indices), num_neg=5)
            
            optimizer.zero_grad()
            loss = model(batch_t, batch_c, batch_neg)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            
    # Extract trained target embeddings
    embeddings = model.w_embeddings.weight.detach().numpy()
    return embeddings, node_order

class LINEModel(nn.Module):
    """
    PyTorch implementation of LINE (Large-scale Information Network Embedding).
    Optimizes:
    - First-Order Proximity: local similarity of directly connected nodes.
    - Second-Order Proximity: similarity of node neighborhoods.
    """
    def __init__(self, num_nodes: int, embedding_dim: int, proximity: str = "first"):
        super().__init__()
        self.proximity = proximity
        
        # Embeddings
        self.u_embeddings = nn.Embedding(num_nodes, embedding_dim)
        if proximity == "second":
            # Context embedding for second-order
            self.c_embeddings = nn.Embedding(num_nodes, embedding_dim)
            
        self.u_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        if proximity == "second":
            self.c_embeddings.weight.data.zero_()
            
    def forward(self, u: torch.Tensor, v: torch.Tensor, neg_samples: torch.Tensor) -> torch.Tensor:
        """Computes proximity loss with negative sampling."""
        embed_u = self.u_embeddings(u)
        
        if self.proximity == "first":
            embed_v = self.u_embeddings(v)
            embed_neg = self.u_embeddings(neg_samples)
        else:
            embed_v = self.c_embeddings(v)
            embed_neg = self.c_embeddings(neg_samples)
            
        # Positive log-sigmoid dot product
        pos_score = torch.sum(embed_u * embed_v, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-15)
        
        # Negative log-sigmoid dot product
        # (batch_size, num_neg_samples, dim) x (batch_size, dim, 1) -> (batch_size, num_neg_samples)
        neg_score = torch.bmm(embed_neg, embed_u.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-15), dim=1)
        
        return -torch.mean(pos_loss + neg_loss)

def train_line(graph: Graph, embedding_dim: int = 16, proximity: str = "first", 
               epochs: int = 10, lr: float = 0.01) -> Tuple[np.ndarray, List[any]]:
    """Trains a LINE embedding model for either first-order or second-order proximity."""
    node_order = list(graph.nodes.keys())
    node_to_idx = {node: i for i, node in enumerate(node_order)}
    n = len(node_order)
    
    # Collect edges
    edge_list = graph.to_edge_list()
    if not edge_list:
        return np.zeros((n, embedding_dim)), node_order
        
    model = LINEModel(n, embedding_dim, proximity)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Form tensors
    u_tensor = torch.tensor([node_to_idx[edge[0]] for edge in edge_list], dtype=torch.long)
    v_tensor = torch.tensor([node_to_idx[edge[1]] for edge in edge_list], dtype=torch.long)
    
    dataset_size = len(edge_list)
    batch_size = 32
    
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_u = u_tensor[indices]
            batch_v = v_tensor[indices]
            batch_neg = torch.randint(0, n, (len(indices), 5))
            
            optimizer.zero_grad()
            loss = model(batch_u, batch_v, batch_neg)
            loss.backward()
            optimizer.step()
            
    embeddings = model.u_embeddings.weight.detach().numpy()
    return embeddings, node_order

def compute_hope(graph: Graph, embedding_dim: int = 16, beta: float = 0.01) -> Tuple[np.ndarray, List[any]]:
    """
    High-Order Proximity preserved Embedding (HOPE) using SVD.
    Preserves Katz Index as similarity: S = (I - beta * A)^(-1) * (beta * A).
    Then uses SVD to decompose: S ≈ U_s * Sigma_s * V_s^T, embedding = U_s * sqrt(Sigma_s).
    """
    A, node_order = graph.to_adjacency_matrix()
    n = len(node_order)
    
    # 1. Compute Katz Index proximity matrix
    I = np.eye(n)
    # S = (I - beta * A)^(-1) * (beta * A)
    try:
        M = I - beta * A
        S = np.dot(la.inv(M), beta * A)
    except la.LinAlgError:
        # Fallback if singular
        S = beta * A
        
    # 2. Perform Singular Value Decomposition
    U, s, Vh = la.svd(S)
    
    # 3. Construct Embeddings (take top k singular values)
    k = min(embedding_dim, n)
    U_k = U[:, :k]
    s_k = s[:k]
    
    # Left and Right embeddings represent source and target directed interactions
    source_embeddings = np.dot(U_k, np.diag(np.sqrt(s_k)))
    
    # Handle padding if k < embedding_dim
    if k < embedding_dim:
        pad = np.zeros((n, embedding_dim - k))
        source_embeddings = np.hstack([source_embeddings, pad])
        
    return source_embeddings, node_order


# ==========================================
# 2. Probabilistic Graphical Models (PGM)
# ==========================================

class BayesianNetwork:
    """
    Represents a Bayesian Network (Directed Graphical Model).
    Manages node conditional probability tables (CPTs) and supports rejection sampling.
    """
    def __init__(self):
        self.dag = Graph(directed=True)
        # cpts[node] = dict: tuple(parent_values) -> dict: {value: probability}
        # Parent values are ordered sorted alphabetically by parent node ID
        self.cpts: Dict[any, Dict[Tuple[any, ...], Dict[any, float]]] = {}
        
    def add_variable(self, node_id: any, parents: List[any], cpt: Dict[Tuple[any, ...], Dict[any, float]]) -> None:
        """Adds a variable with its parents and CPT."""
        self.dag.add_node(node_id)
        for p in parents:
            self.dag.add_edge(p, node_id)
        self.cpts[node_id] = cpt
        
    def get_parents(self, node_id: any) -> List[any]:
        """Returns sorted list of parent nodes for node_id."""
        return sorted(list(self.dag.rev_adj[node_id].keys()))
        
    def sample_joint(self) -> Dict[any, any]:
        """Generates a single joint sample using Forward (Ancestral) Sampling."""
        # Topologically sort nodes to sample parents before children
        from src.phase2_algorithms import topological_sort
        sorted_nodes = topological_sort(self.dag)
        
        sample = {}
        for node in sorted_nodes:
            parents = self.get_parents(node)
            # Fetch parent values for CPT lookup
            parent_vals = tuple(sample[p] for p in parents)
            
            # Get probability distribution
            prob_dist = self.cpts[node][parent_vals]
            
            # Sample value
            choices = list(prob_dist.keys())
            probs = list(prob_dist.values())
            val = np.random.choice(choices, p=probs)
            sample[node] = val
            
        return sample

    def rejection_sampling(self, query_var: any, evidence: Dict[any, any], num_samples: int = 1000) -> Dict[any, float]:
        """
        Estimates posterior distribution P(QueryVar | Evidence) using Rejection Sampling.
        Rejects joint samples that do not match the specified evidence.
        """
        counts = {}
        accepted_samples = 0
        
        for _ in range(num_samples):
            sample = self.sample_joint()
            
            # Check evidence match
            matches_evidence = True
            for ev_var, ev_val in evidence.items():
                if sample[ev_var] != ev_val:
                    matches_evidence = False
                    break
                    
            if matches_evidence:
                val = sample[query_var]
                counts[val] = counts.get(val, 0) + 1
                accepted_samples += 1
                
        if accepted_samples == 0:
            return {} # Avoid divide-by-zero if evidence was extremely rare
            
        return {val: count / accepted_samples for val, count in counts.items()}

class FactorGraph:
    """
    Represents a Factor Graph (undirected graphical model with variables and factor nodes).
    Supports belief propagation (sum-product) on tree architectures.
    """
    def __init__(self):
        self.variables: Set[any] = set()
        # factors[factor_name] = (list_of_scope_vars, numpy_potential_table)
        self.factors: Dict[str, Tuple[List[any], np.ndarray]] = {}
        # Neighbor connections: node -> set of connected factor/var nodes
        self.neighbors: Dict[any, Set[any]] = {}
        
    def add_variable(self, var_id: any) -> None:
        self.variables.add(var_id)
        if var_id not in self.neighbors:
            self.neighbors[var_id] = set()
            
    def add_factor(self, factor_name: str, scope: List[any], potentials: np.ndarray) -> None:
        """Adds a factor with its scope variables and potential table (numpy array)."""
        self.factors[factor_name] = (scope, potentials)
        if factor_name not in self.neighbors:
            self.neighbors[factor_name] = set()
            
        for var in scope:
            self.add_variable(var)
            self.neighbors[var].add(factor_name)
            self.neighbors[factor_name].add(var)

    def run_belief_propagation(self, max_iter: int = 10) -> Dict[any, np.ndarray]:
        """
        Runs Loopy Belief Propagation (Sum-Product Message Passing).
        Returns marginal belief arrays for all variables.
        """
        # Messages dictionary: (source, target) -> numpy array of message values
        messages = {}
        
        # Initialize messages: 
        # Variable-to-Factor message (uniform ones)
        # Factor-to-Variable message (uniform ones or marginalized factor potential)
        for node in self.neighbors:
            is_var = node in self.variables
            for neighbor in self.neighbors[node]:
                # Assume binary variables for simplicity in this educational script (cardinality 2)
                messages[(node, neighbor)] = np.ones(2)
                
        # Message updating loop
        for _ in range(max_iter):
            next_messages = {}
            
            for (src, dst) in messages:
                if src in self.variables:
                    # Variable to Factor Message: Product of incoming messages from all OTHER factors
                    incoming_factors = self.neighbors[src] - {dst}
                    msg = np.ones(2)
                    for f in incoming_factors:
                        msg *= messages[(f, src)]
                    # Normalize message
                    msg_sum = np.sum(msg)
                    if msg_sum > 0:
                        msg /= msg_sum
                    next_messages[(src, dst)] = msg
                else:
                    # Factor to Variable Message: Sum-Product algorithm
                    scope, potentials = self.factors[src]
                    dst_idx = scope.index(dst)
                    
                    # We need to compute: sum_{X \ {dst}} phi(X) * prod_{y \in scope \ {dst}} m_{y -> factor}
                    # Construct message update
                    # For educational brevity in a 2-variable factor:
                    if len(scope) == 2:
                        other_var = scope[0] if scope[1] == dst else scope[1]
                        other_idx = scope.index(other_var)
                        other_msg = messages[(other_var, src)]
                        
                        # Sum product: marginalize out other variable
                        # potentials shape is (card(scope[0]), card(scope[1]))
                        msg = np.zeros(2)
                        for dst_val in range(2):
                            val_sum = 0.0
                            for other_val in range(2):
                                idxs = [0, 0]
                                idxs[dst_idx] = dst_val
                                idxs[other_idx] = other_val
                                val_sum += potentials[tuple(idxs)] * other_msg[other_val]
                            msg[dst_val] = val_sum
                            
                        # Normalize
                        msg_sum = np.sum(msg)
                        if msg_sum > 0:
                            msg /= msg_sum
                        next_messages[(src, dst)] = msg
                    else:
                        # Fallback simple potential copy for unary factors
                        msg = potentials.copy()
                        msg_sum = np.sum(msg)
                        if msg_sum > 0:
                            msg /= msg_sum
                        next_messages[(src, dst)] = msg
                        
            messages = next_messages
            
        # Calculate final marginal beliefs
        beliefs = {}
        for var in self.variables:
            b = np.ones(2)
            for f in self.neighbors[var]:
                b *= messages[(f, var)]
            b_sum = np.sum(b)
            if b_sum > 0:
                b /= b_sum
            beliefs[var] = b
            
        return beliefs

class HiddenMarkovModel:
    """
    Hidden Markov Model (HMM) representing sequential probabilistic dependencies.
    Implements the classic Viterbi Decoding algorithm.
    """
    def __init__(self, states: List[any], emissions: List[any], 
                 transition_matrix: np.ndarray, emission_matrix: np.ndarray, 
                 initial_probs: np.ndarray):
        self.states = states
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.emissions = emissions
        self.emission_to_idx = {e: i for i, e in enumerate(emissions)}
        
        self.A = transition_matrix      # States x States transition matrix
        self.B = emission_matrix        # States x Emissions emission matrix
        self.pi = initial_probs          # Prior state probabilities
        
    def viterbi(self, obs_sequence: List[any]) -> Tuple[List[any], float]:
        """
        Viterbi Algorithm (Dynamic Programming, O(T * S^2)) to decode the most likely
        sequence of hidden states given a sequence of observations.
        """
        T = len(obs_sequence)
        num_states = len(self.states)
        
        # dp[t, s] stores maximum probability of path ending in state s at time t
        dp = np.zeros((T, num_states))
        # backptr[t, s] stores predecessor state at time t-1 leading to max prob for state s
        backptr = np.zeros((T, num_states), dtype=int)
        
        # Initialize time step 0
        first_obs_idx = self.emission_to_idx[obs_sequence[0]]
        for s in range(num_states):
            dp[0, s] = self.pi[s] * self.B[s, first_obs_idx]
            
        # Dynamic programming transitions
        for t in range(1, T):
            obs_idx = self.emission_to_idx[obs_sequence[t]]
            for s in range(num_states):
                # Calculate max probability from any previous state prev_s to s
                probs = [dp[t-1, prev_s] * self.A[prev_s, s] * self.B[s, obs_idx] for prev_s in range(num_states)]
                dp[t, s] = max(probs)
                backptr[t, s] = np.argmax(probs)
                
        # Backtracking path reconstruction
        best_last_state = np.argmax(dp[T-1, :])
        max_prob = dp[T-1, best_last_state]
        
        state_path = [self.states[best_last_state]]
        curr_state_idx = best_last_state
        for t in range(T-1, 0, -1):
            curr_state_idx = backptr[t, curr_state_idx]
            state_path.insert(0, self.states[curr_state_idx])
            
        return state_path, float(max_prob)

class LinearChainCRF:
    """
    Lightweight Linear-Chain Conditional Random Field (CRF) decoder.
    Computes most likely label sequence using feature potential functions and Viterbi.
    """
    def __init__(self, labels: List[any]):
        self.labels = labels
        self.num_labels = len(labels)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        
    def decode(self, sentence_len: int, emission_potentials: np.ndarray, transition_potentials: np.ndarray) -> List[any]:
        """
        Decodes sequence using Viterbi.
        emission_potentials: T x L matrix representing state feature scores.
        transition_potentials: L x L matrix representing transition scores from Label_i to Label_j.
        """
        T = sentence_len
        L = self.num_labels
        
        # Viterbi tables in log-space
        dp = np.full((T, L), -float('inf'))
        backptr = np.zeros((T, L), dtype=int)
        
        # Initial step
        for l in range(L):
            dp[0, l] = emission_potentials[0, l]
            
        # Recursion
        for t in range(1, T):
            for l in range(L):
                scores = [dp[t-1, prev_l] + transition_potentials[prev_l, l] + emission_potentials[t, l] for prev_l in range(L)]
                dp[t, l] = max(scores)
                backptr[t, l] = np.argmax(scores)
                
        # Backtrack
        best_last = np.argmax(dp[T-1, :])
        path = [self.labels[best_last]]
        curr = best_last
        for t in range(T-1, 0, -1):
            curr = backptr[t, curr]
            path.insert(0, self.labels[curr])
            
        return path
