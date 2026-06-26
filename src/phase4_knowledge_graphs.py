"""
Graph Ecosystem Masterclass - Phase 4: Knowledge Graphs, Query Languages & KGEs
=============================================================================
This module implements Knowledge Graph (KG) architectures from first principles.
It contains an in-memory Triple Store with full indexes (SPO, POS, OSP) supporting
SPARQL-like pattern matching queries, Entity Resolution, and 7 major KGE models
(TransE, TransH, TransR, RotatE, DistMult, ComplEx, RESCAL) implemented in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set, Union, Optional, Callable
from src.phase1_foundations import Graph

# ==========================================
# 1. Triple Store & SPARQL-Like Interpreter
# ==========================================

class TripleStore:
    """
    An in-memory Semantic Triple Store (RDF-like) featuring three index registries
    (SPO, POS, OSP) for O(1) query matching and pattern search.
    
    Triples are stored as: (Subject, Predicate, Object).
    """
    def __init__(self):
        self.triples: Set[Tuple[str, str, str]] = set()
        
        # Index Registries
        self.spo: Dict[str, Dict[str, Set[str]]] = {}  # Subject -> Predicate -> Objects
        self.pos: Dict[str, Dict[str, Set[str]]] = {}  # Predicate -> Object -> Subjects
        self.osp: Dict[str, Dict[str, Set[str]]] = {}  # Object -> Subject -> Predicates
        
    def add_triple(self, s: str, p: str, o: str) -> None:
        """Adds a triple and updates SPO, POS, and OSP index registries."""
        triple = (s, p, o)
        if triple in self.triples:
            return
        self.triples.add(triple)
        
        # Update SPO index
        if s not in self.spo: self.spo[s] = {}
        if p not in self.spo[s]: self.spo[s][p] = set()
        self.spo[s][p].add(o)
        
        # Update POS index
        if p not in self.pos: self.pos[p] = {}
        if o not in self.pos[p]: self.pos[p][o] = set()
        self.pos[p][o].add(s)
        
        # Update OSP index
        if o not in self.osp: self.osp[o] = {}
        if s not in self.osp[o]: self.osp[o][s] = set()
        self.osp[o][s].add(p)
        
    def query_single_pattern(self, s: str, p: str, o: str) -> Set[Tuple[str, str, str]]:
        """
        Queries the store for a single triple pattern.
        Arguments can be concrete strings or variables starting with "?" (e.g., "?x").
        Utilizes index registries to achieve O(1) retrieval.
        """
        is_var_s = s.startswith("?")
        is_var_p = p.startswith("?")
        is_var_o = o.startswith("?")
        
        results = set()
        
        # Case 1: All variables (?s, ?p, ?o)
        if is_var_s and is_var_p and is_var_o:
            return self.triples
            
        # Case 2: Subject is constant, Predicate/Object are variables (s, ?p, ?o) -> use SPO index
        elif not is_var_s and is_var_p and is_var_o:
            if s in self.spo:
                for pred in self.spo[s]:
                    for obj in self.spo[s][pred]:
                        results.add((s, pred, obj))
                        
        # Case 3: Subject and Predicate are constants, Object is variable (s, p, ?o) -> use SPO index
        elif not is_var_s and not is_var_p and is_var_o:
            if s in self.spo and p in self.spo[s]:
                for obj in self.spo[s][p]:
                    results.add((s, p, obj))
                    
        # Case 4: Predicate is constant, Subject/Object are variables (?s, p, ?o) -> use POS index
        elif is_var_s and not is_var_p and is_var_o:
            if p in self.pos:
                for obj in self.pos[p]:
                    for subj in self.pos[p][obj]:
                        results.add((subj, p, obj))
                        
        # Case 5: Predicate and Object are constants, Subject is variable (?s, p, o) -> use POS index
        elif is_var_s and not is_var_p and not is_var_o:
            if p in self.pos and o in self.pos[p]:
                for subj in self.pos[p][o]:
                    results.add((subj, p, o))
                    
        # Case 6: Object is constant, Subject/Predicate are variables (?s, ?p, o) -> use OSP index
        elif is_var_s and is_var_p and not is_var_o:
            if o in self.osp:
                for subj in self.osp[o]:
                    for pred in self.osp[o][subj]:
                        results.add((subj, pred, o))
                        
        # Case 7: Subject and Object are constants, Predicate is variable (s, ?p, o) -> use OSP index
        elif not is_var_s and is_var_p and not is_var_o:
            if o in self.osp and s in self.osp[o]:
                for pred in self.osp[o][s]:
                    results.add((s, pred, o))
                    
        # Case 8: Fully bound query (s, p, o)
        else:
            if (s, p, o) in self.triples:
                results.add((s, p, o))
                
        return results

    def query(self, patterns: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """
        Executes a SPARQL-like multi-pattern query (Basic Graph Pattern join).
        Supports variable binding joins across multiple hops.
        
        Args:
            patterns: List of triple patterns, e.g. [("?person", "works_at", "?company"), ("?company", "located_in", "USA")]
            
        Returns:
            bindings: List of dictionaries mapping variable names to entity values.
        """
        if not patterns:
            return []
            
        # Recursive backtracking solver to bind variables
        results: List[Dict[str, str]] = []
        
        # Extract all variable names from patterns
        variables = set()
        for s, p, o in patterns:
            if s.startswith("?"): variables.add(s)
            if p.startswith("?"): variables.add(p)
            if o.startswith("?"): variables.add(o)
            
        variables_list = list(variables)
        
        def backtrack(pattern_idx: int, current_bindings: Dict[str, str]):
            if pattern_idx == len(patterns):
                # All patterns matched, save bindings
                results.append(current_bindings.copy())
                return
                
            # Fetch current pattern
            s_pat, p_pat, o_pat = patterns[pattern_idx]
            
            # Replace bound variables in pattern
            s_val = current_bindings.get(s_pat, s_pat)
            p_val = current_bindings.get(p_pat, p_pat)
            o_val = current_bindings.get(o_pat, o_pat)
            
            # Query matching triples for this pattern
            matching_triples = self.query_single_pattern(s_val, p_val, o_val)
            
            for s_match, p_match, o_match in matching_triples:
                # Try to bind variables
                new_bindings = current_bindings.copy()
                conflict = False
                
                # Check Subject binding
                if s_pat.startswith("?"):
                    if s_pat in new_bindings and new_bindings[s_pat] != s_match:
                        conflict = True
                    else:
                        new_bindings[s_pat] = s_match
                        
                # Check Predicate binding
                if p_pat.startswith("?"):
                    if p_pat in new_bindings and new_bindings[p_pat] != p_match:
                        conflict = True
                    else:
                        new_bindings[p_pat] = p_match
                        
                # Check Object binding
                if o_pat.startswith("?"):
                    if o_pat in new_bindings and new_bindings[o_pat] != o_match:
                        conflict = True
                    else:
                        new_bindings[o_pat] = o_match
                        
                if not conflict:
                    backtrack(pattern_idx + 1, new_bindings)

        backtrack(0, {})
        return results


# ==========================================
# 2. Graph Resolution & Integration
# ==========================================

class GraphResolver:
    """Implements Entity Linking and Entity Resolution algorithms for KG integration."""
    
    @staticmethod
    def entity_linking(mention: str, candidate_entities: Dict[str, str], threshold: float = 0.6) -> Optional[str]:
        """
        Links a textual mention to a candidate KG entity using Levenshtein edit distance.
        """
        def edit_distance(s1: str, s2: str) -> int:
            m, n = len(s1), len(s2)
            dp = np.zeros((m + 1, n + 1))
            for i in range(m + 1): dp[i, 0] = i
            for j in range(n + 1): dp[0, j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if s1[i-1] == s2[j-1] else 1
                    dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
            return int(dp[m, n])
            
        best_entity = None
        min_dist = float('inf')
        
        for entity_id, entity_name in candidate_entities.items():
            dist = edit_distance(mention.lower(), entity_name.lower())
            max_len = max(len(mention), len(entity_name))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0
            
            if sim >= threshold and dist < min_dist:
                min_dist = dist
                best_entity = entity_id
                
        return best_entity

    @staticmethod
    def entity_resolution(graph: Graph, node_a: any, node_b: any, attribute_sim_fn: Callable = None, threshold: float = 0.7) -> bool:
        """
        Deduplicates nodes in a graph using attributes and Neighborhood Jaccard similarity.
        Jaccard(N(A), N(B)) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
        """
        # 1. Neighborhood similarity
        neighbors_a = set(graph.adj[node_a].keys())
        neighbors_b = set(graph.adj[node_b].keys())
        
        intersection = len(neighbors_a.intersection(neighbors_b))
        union = len(neighbors_a.union(neighbors_b))
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # 2. Attribute similarity
        attr_sim = 1.0
        if attribute_sim_fn is not None:
            attrs_a = graph.nodes.get(node_a, {})
            attrs_b = graph.nodes.get(node_b, {})
            attr_sim = attribute_sim_fn(attrs_a, attrs_b)
            
        # Combine similarities (weighted average)
        combined_sim = 0.4 * jaccard_sim + 0.6 * attr_sim
        return combined_sim >= threshold


# ==========================================
# 3. Knowledge Graph Embeddings (KGE)
# ==========================================

class TransE(nn.Module):
    """
    TransE: h + r ≈ t.
    Scoring function: f_r(h, t) = - || h + r - t ||_{L1 or L2}
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int, norm: int = 1):
        super().__init__()
        self.norm = norm
        self.ent_embeddings = nn.Embedding(num_entities, dim)
        self.rel_embeddings = nn.Embedding(num_relations, dim)
        
        # Initialize according to uniform distribution
        self.ent_embeddings.weight.data.uniform_(-6 / np.sqrt(dim), 6 / np.sqrt(dim))
        self.rel_embeddings.weight.data.uniform_(-6 / np.sqrt(dim), 6 / np.sqrt(dim))
        # Relation normalisation
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(relations)
        t = self.ent_embeddings(tails)
        
        # Normalise entity embeddings (TransE constraint: ||e||_2 <= 1)
        h = F.normalize(h, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        
        # Distance calculation
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

class TransH(nn.Module):
    """
    TransH: Project entities onto relation-specific hyperplane: h_p = h - w_r^T h w_r.
    Then translate: h_p + d_r ≈ t_p.
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, dim)
        self.rel_embeddings = nn.Embedding(num_relations, dim) # translation vector d_r
        self.norm_vectors = nn.Embedding(num_relations, dim)   # hyperplane norm w_r
        
        self.ent_embeddings.weight.data.uniform_(-6 / np.sqrt(dim), 6 / np.sqrt(dim))
        self.rel_embeddings.weight.data.uniform_(-6 / np.sqrt(dim), 6 / np.sqrt(dim))
        self.norm_vectors.weight.data.uniform_(-6 / np.sqrt(dim), 6 / np.sqrt(dim))

    def _project(self, e: torch.Tensor, w_r: torch.Tensor) -> torch.Tensor:
        # e: (batch, dim), w_r: (batch, dim)
        # Projected e = e - (w_r^T * e) * w_r
        dot = torch.sum(e * w_r, dim=1, keepdim=True) # (batch, 1)
        return e - dot * w_r

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_embeddings(heads)
        t = self.ent_embeddings(tails)
        
        d_r = self.rel_embeddings(relations)
        w_r = F.normalize(self.norm_vectors(relations), p=2, dim=1) # Ensure w_r has unit length
        
        h_proj = self._project(h, w_r)
        t_proj = self._project(t, w_r)
        
        score = torch.norm(h_proj + d_r - t_proj, p=1, dim=1)
        return score

class TransR(nn.Module):
    """
    TransR: Project entities into relation-specific spaces using a projection matrix M_r.
    h_r = h * M_r, t_r = t * M_r. Translation: h_r + r ≈ t_r.
    """
    def __init__(self, num_entities: int, num_relations: int, ent_dim: int, rel_dim: int):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, ent_dim)
        self.rel_embeddings = nn.Embedding(num_relations, rel_dim)
        # Projection matrices: Relation -> Matrix of size (ent_dim x rel_dim)
        self.proj_matrices = nn.Parameter(torch.Tensor(num_relations, ent_dim, rel_dim))
        
        self.ent_embeddings.weight.data.uniform_(-6 / np.sqrt(ent_dim), 6 / np.sqrt(ent_dim))
        self.rel_embeddings.weight.data.uniform_(-6 / np.sqrt(rel_dim), 6 / np.sqrt(rel_dim))
        nn.init.xavier_uniform_(self.proj_matrices)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_embeddings(heads).unsqueeze(1) # (batch, 1, ent_dim)
        t = self.ent_embeddings(tails).unsqueeze(1) # (batch, 1, ent_dim)
        
        r = self.rel_embeddings(relations) # (batch, rel_dim)
        M_r = self.proj_matrices[relations] # (batch, ent_dim, rel_dim)
        
        # Project: (batch, 1, ent_dim) x (batch, ent_dim, rel_dim) -> (batch, 1, rel_dim) -> squeeze -> (batch, rel_dim)
        h_proj = torch.bmm(h, M_r).squeeze(1)
        t_proj = torch.bmm(t, M_r).squeeze(1)
        
        score = torch.norm(h_proj + r - t_proj, p=1, dim=1)
        return score

class RotatE(nn.Module):
    """
    RotatE: Define relations as rotations in complex vector space.
    t = h ∘ r, where h, t, r are complex, and |r_i| = 1.
    Score: - || h ∘ r - t ||
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.dim = dim
        # Complex numbers represented by splitting embedding dimension in two (real, imaginary parts)
        self.ent_real = nn.Embedding(num_entities, dim)
        self.ent_imag = nn.Embedding(num_entities, dim)
        
        # Relations are phases (theta) in range [0, 2pi] representing rotation
        self.rel_phase = nn.Embedding(num_relations, dim)
        
        self.ent_real.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.ent_imag.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.rel_phase.weight.data.uniform_(-np.pi, np.pi)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h_re = self.ent_real(heads)
        h_im = self.ent_imag(heads)
        t_re = self.ent_real(tails)
        t_im = self.ent_imag(tails)
        
        # Relation rotation represented as Euler's formula: r = cos(theta) + i*sin(theta)
        r_phase = self.rel_phase(relations)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        
        # Complex multiplication: h * r = (h_re * r_re - h_im * r_im) + i * (h_re * r_im + h_im * r_re)
        prod_re = h_re * r_re - h_im * r_im
        prod_im = h_re * r_im + h_im * r_re
        
        # Distance in complex space: || (h * r) - t ||
        re_diff = prod_re - t_re
        im_diff = prod_im - t_im
        
        score = torch.sum(re_diff**2 + im_diff**2, dim=1)
        return torch.sqrt(score + 1e-15)

class DistMult(nn.Module):
    """
    DistMult: Bilinear model with diagonal relation matrices.
    Scoring function: f(h, r, t) = h^T * diag(M_r) * t
    Scoring is symmetric: f(h, r, t) == f(t, r, h).
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, dim)
        self.rel_embeddings = nn.Embedding(num_relations, dim)
        
        self.ent_embeddings.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.rel_embeddings.weight.data.uniform_(-0.5 / dim, 0.5 / dim)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(relations)
        t = self.ent_embeddings(tails)
        
        # Score = sum(h_i * r_i * t_i)
        # Higher score means higher probability. In our loss we will maximize this score.
        score = torch.sum(h * r * t, dim=1)
        return score

class ComplEx(nn.Module):
    """
    ComplEx: Bilinear model in complex vector space (breaks symmetry of DistMult).
    Scoring function: Re(h^T * diag(r) * conj(t))
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        # Complex coordinates
        self.ent_real = nn.Embedding(num_entities, dim)
        self.ent_imag = nn.Embedding(num_entities, dim)
        self.rel_real = nn.Embedding(num_relations, dim)
        self.rel_imag = nn.Embedding(num_relations, dim)
        
        self.ent_real.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.ent_imag.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.rel_real.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.rel_imag.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h_re = self.ent_real(heads)
        h_im = self.ent_imag(heads)
        r_re = self.rel_real(relations)
        r_im = self.rel_imag(relations)
        t_re = self.ent_real(tails)
        t_im = self.ent_imag(tails)
        
        # Real part of: h * r * conj(t)
        # = Re( (h_re + i*h_im) * (r_re + i*r_im) * (t_re - i*t_im) )
        # = (h_re*r_re - h_im*r_im)*t_re + (h_re*r_im + h_im*r_re)*t_im
        score = torch.sum(
            (h_re * r_re - h_im * r_im) * t_re +
            (h_re * r_im + h_im * r_re) * t_im,
            dim=1
        )
        return score

class RESCAL(nn.Module):
    """
    RESCAL: Bilinear tensor factorization using a full relation matrix.
    Scoring function: f(h, r, t) = h^T * M_r * t
    """
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, dim)
        # Full bilinear relation matrices (num_relations x dim x dim)
        self.rel_matrices = nn.Parameter(torch.Tensor(num_relations, dim, dim))
        
        self.ent_embeddings.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        nn.init.xavier_uniform_(self.rel_matrices)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_embeddings(heads).unsqueeze(1) # (batch, 1, dim)
        t = self.ent_embeddings(tails).unsqueeze(2) # (batch, dim, 1)
        
        M_r = self.rel_matrices[relations] # (batch, dim, dim)
        
        # Score = h * M_r * t -> (batch, 1, dim) x (batch, dim, dim) x (batch, dim, 1) -> (batch, 1, 1) -> squeeze
        score = torch.bmm(torch.bmm(h, M_r), t).squeeze(2).squeeze(1)
        return score

class KGETrainer:
    """
    Trainer engine for Knowledge Graph Embeddings.
    Uses pairwise margin ranking loss: Loss = max(0, margin + d(pos) - d(neg))
    or standard margin ranking binary cross-entropy depending on scoring format.
    
    For Distance-based models (TransE, TransH, TransR, RotatE), we minimize the score.
    For Similarity-based models (DistMult, ComplEx, RESCAL), we maximize the score.
    """
    def __init__(self, model: nn.Module, model_type: str, margin: float = 1.0, lr: float = 0.01):
        self.model = model
        self.model_type = model_type
        self.margin = margin
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, pos_h: List[int], pos_r: List[int], pos_t: List[int], 
                   neg_h: List[int], neg_r: List[int], neg_t: List[int]) -> float:
        """Runs a single training step on a batch of positive and negative triples."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Convert to Tensors
        p_h = torch.tensor(pos_h, dtype=torch.long)
        p_r = torch.tensor(pos_r, dtype=torch.long)
        p_t = torch.tensor(pos_t, dtype=torch.long)
        
        n_h = torch.tensor(neg_h, dtype=torch.long)
        n_r = torch.tensor(neg_r, dtype=torch.long)
        n_t = torch.tensor(neg_t, dtype=torch.long)
        
        pos_scores = self.model(p_h, p_r, p_t)
        neg_scores = self.model(n_h, n_r, n_t)
        
        # Compute Margin-based pairwise ranking loss
        # Distance-based: pos should have SMALL distance, neg should have LARGE distance.
        # Loss = max(0, margin + pos_dist - neg_dist)
        # Similarity-based: pos should have LARGE similarity, neg should have SMALL similarity.
        # Loss = max(0, margin - pos_sim + neg_sim)
        distance_based = self.model_type in ["transe", "transh", "transr", "rotate"]
        
        if distance_based:
            loss = torch.mean(F.relu(self.margin + pos_scores - neg_scores))
        else:
            loss = torch.mean(F.relu(self.margin - pos_scores + neg_scores))
            
        loss.backward()
        self.optimizer.step()
        return loss.item()
