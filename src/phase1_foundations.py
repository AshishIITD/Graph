"""
Graph Ecosystem Masterclass - Phase 1: Graph Foundations, Representation & Spectral Theory
========================================================================================
This module implements the core Graph data structures, representations, topological properties,
centrality measures, and Spectral Graph Theory foundations from first principles.

Every concept is heavily commented to serve as a high-quality educational resource.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Set, Union, Optional

class Graph:
    """
    A versatile, from-scratch Graph data structure supporting:
    - Directed & Undirected edges
    - Weighted & Unweighted connections
    - Heterogeneous & Homogeneous nodes and edges
    - Multi-graphs and Hypergraphs (via hyperedge-to-node mappings)
    
    Attributes:
        directed (bool): True if edges have direction, False otherwise.
        nodes (dict): Mapping of node_id -> dict of properties (including 'type').
        adj (dict): Adjacency list representation: u -> v -> dict of edge properties (weighted, etc.).
        rev_adj (dict): Reverse adjacency list (for directed graphs, tracing in-edges).
    """
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.nodes: Dict[any, Dict[str, any]] = {}
        self.adj: Dict[any, Dict[any, Dict[str, any]]] = {}
        self.rev_adj: Dict[any, Dict[any, Dict[str, any]]] = {}
        
    def add_node(self, node_id: any, node_type: str = "default", **properties) -> None:
        """Adds a node with arbitrary metadata properties to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, **properties}
            self.adj[node_id] = {}
            self.rev_adj[node_id] = {}
        else:
            self.nodes[node_id].update(properties)
            
    def add_edge(self, u: any, v: any, weight: float = 1.0, edge_type: str = "default", **properties) -> None:
        """
        Adds an edge between u and v. Automatically adds nodes if they don't exist.
        If undirected, adds the edge in both directions.
        """
        self.add_node(u)
        self.add_node(v)
        
        edge_data = {"weight": weight, "type": edge_type, **properties}
        
        self.adj[u][v] = edge_data
        self.rev_adj[v][u] = edge_data
        
        if not self.directed:
            self.adj[v][u] = edge_data
            self.rev_adj[u][v] = edge_data
            
    def remove_edge(self, u: any, v: any) -> None:
        """Removes the edge between u and v."""
        if u in self.adj and v in self.adj[u]:
            del self.adj[u][v]
        if v in self.rev_adj and u in self.rev_adj[v]:
            del self.rev_adj[v][u]
            
        if not self.directed:
            if v in self.adj and u in self.adj[v]:
                del self.adj[v][u]
            if u in self.rev_adj and v in self.rev_adj[u]:
                del self.rev_adj[u][v]
                
    def remove_node(self, node_id: any) -> None:
        """Removes a node and all of its incident edges."""
        if node_id in self.nodes:
            # Remove all outgoing edges
            targets = list(self.adj[node_id].keys())
            for v in targets:
                self.remove_edge(node_id, v)
                
            # Remove all incoming edges
            sources = list(self.rev_adj[node_id].keys())
            for u in sources:
                self.remove_edge(u, node_id)
                
            del self.nodes[node_id]
            del self.adj[node_id]
            del self.rev_adj[node_id]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
        
    @property
    def num_edges(self) -> int:
        count = sum(len(self.adj[u]) for u in self.adj)
        return count if self.directed else count // 2

    # ==========================================
    # 2. Graph Representations
    # ==========================================
    
    def to_edge_list(self) -> List[Tuple[any, any, float]]:
        """Returns the graph as a list of triples: (u, v, weight)."""
        edges = []
        seen = set()
        for u in self.adj:
            for v, data in self.adj[u].items():
                if self.directed:
                    edges.append((u, v, data["weight"]))
                else:
                    edge_key = tuple(sorted([str(u), str(v)]))
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append((u, v, data["weight"]))
        return edges

    def to_adjacency_list(self) -> Dict[any, List[Tuple[any, float]]]:
        """Returns the graph as an adjacency list: node -> list of (neighbor, weight)."""
        adj_list = {}
        for u in self.adj:
            adj_list[u] = [(v, data["weight"]) for v, data in self.adj[u].items()]
        return adj_list

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[any]]:
        """
        Converts the graph into a dense 2D Adjacency Matrix.
        Returns:
            matrix (np.ndarray): N x N matrix where entry (i, j) is the edge weight (0 if no edge).
            node_order (list): List of node IDs corresponding to matrix indices.
        """
        node_order = list(self.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_order)}
        n = len(node_order)
        matrix = np.zeros((n, n))
        
        for u in self.adj:
            u_idx = node_to_idx[u]
            for v, data in self.adj[u].items():
                v_idx = node_to_idx[v]
                matrix[u_idx, v_idx] = data["weight"]
                
        return matrix, node_order

    def to_incidence_matrix(self) -> Tuple[np.ndarray, List[any], List[Tuple[any, any]]]:
        """
        Converts the graph into an Incidence Matrix (Nodes x Edges).
        For undirected graphs:
            Entry (i, e) is 1 if node i is incident to edge e, 0 otherwise.
        For directed graphs:
            Entry (i, e) is -1 if edge e leaves node i, +1 if edge e enters node i, 0 otherwise.
        """
        node_order = list(self.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_order)}
        edge_list = self.to_edge_list()
        
        num_nodes = len(node_order)
        num_edges = len(edge_list)
        
        matrix = np.zeros((num_nodes, num_edges))
        
        for e_idx, (u, v, _) in enumerate(edge_list):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            if self.directed:
                matrix[u_idx, e_idx] = -1  # Source
                matrix[v_idx, e_idx] = 1   # Target
            else:
                matrix[u_idx, e_idx] = 1
                matrix[v_idx, e_idx] = 1
                
        return matrix, node_order, [(u, v) for u, v, _ in edge_list]

    def to_sparse_matrix(self) -> Tuple[sp.coo_matrix, List[any]]:
        """Returns the Adjacency Matrix represented as a SciPy Sparse Coordinate Matrix."""
        adj_matrix, node_order = self.to_adjacency_matrix()
        return sp.coo_matrix(adj_matrix), node_order

    @classmethod
    def from_adjacency_matrix(cls, matrix: np.ndarray, node_order: Optional[List[any]] = None, directed: bool = False) -> 'Graph':
        """Constructs a Graph from a 2D Adjacency Matrix."""
        n = matrix.shape[0]
        if node_order is None:
            node_order = list(range(n))
            
        g = cls(directed=directed)
        for i in range(n):
            g.add_node(node_order[i])
            
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val != 0:
                    # If undirected, only add once to avoid redundant updates, 
                    # but our add_edge handles it gracefully anyway.
                    g.add_edge(node_order[i], node_order[j], weight=float(val))
        return g


    # ==========================================
    # 3. Graph Properties & Metrics
    # ==========================================
    
    def degree(self, node_id: any) -> Union[int, Tuple[int, int]]:
        """
        Returns the degree of a node.
        If undirected: returns total degree (int).
        If directed: returns tuple (in_degree, out_degree).
        """
        if node_id not in self.nodes:
            raise ValueError("Node not in graph")
            
        out_deg = len(self.adj[node_id])
        in_deg = len(self.rev_adj[node_id])
        
        if self.directed:
            return in_deg, out_deg
        return out_deg

    def density(self) -> float:
        """
        Computes graph density.
        Undirected density: 2 * |E| / (|V| * (|V| - 1))
        Directed density: |E| / (|V| * (|V| - 1))
        """
        n = self.num_nodes
        if n <= 1:
            return 0.0
        e = self.num_edges
        possible_edges = n * (n - 1)
        if self.directed:
            return e / possible_edges
        return (2 * e) / possible_edges

    def clustering_coefficient(self, node_id: any) -> float:
        """
        Computes the local clustering coefficient of a node:
        The fraction of pairs of neighbors that are connected to each other.
        """
        if node_id not in self.nodes:
            raise ValueError("Node not in graph")
            
        neighbors = set(self.adj[node_id].keys())
        # In directed graphs, combine out-neighbors and in-neighbors
        if self.directed:
            neighbors = neighbors.union(self.rev_adj[node_id].keys())
            
        # Exclude the node itself
        neighbors.discard(node_id)
        
        k = len(neighbors)
        if k <= 1:
            return 0.0
            
        # Count links between neighbors
        links = 0
        for u in neighbors:
            for v in neighbors:
                if v in self.adj[u]:
                    links += 1
                    
        # If undirected, each link between neighbors is counted twice (u->v and v->u)
        possible_links = k * (k - 1)
        if not self.directed:
            return (links / 2) / (possible_links / 2)
        return links / possible_links

    def global_clustering_coefficient(self) -> float:
        """
        Computes the average local clustering coefficient across all nodes.
        Also known as clustering coefficient of the graph.
        """
        if self.num_nodes == 0:
            return 0.0
        coeffs = [self.clustering_coefficient(node) for node in self.nodes]
        return sum(coeffs) / len(coeffs)

    def get_shortest_paths_bfs(self, start_node: any) -> Dict[any, float]:
        """Runs BFS to find the shortest unweighted path lengths from start_node."""
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        queue = [start_node]
        head = 0
        
        while head < len(queue):
            curr = queue[head]
            head += 1
            for neighbor in self.adj[curr]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances

    def compute_all_pairs_shortest_paths_unweighted(self) -> Dict[any, Dict[any, float]]:
        """Computes shortest paths between all pairs of nodes using BFS (unweighted)."""
        return {node: self.get_shortest_paths_bfs(node) for node in self.nodes}

    def ecc_rad_diam(self) -> Tuple[Dict[any, float], float, float]:
        """
        Computes topological properties based on unweighted geodesic distance:
        - Eccentricity: Maximum distance from a node to any other node.
        - Radius: Minimum eccentricity in the graph.
        - Diameter: Maximum eccentricity in the graph.
        """
        if self.num_nodes == 0:
            return {}, 0.0, 0.0
            
        apsp = self.compute_all_pairs_shortest_paths_unweighted()
        eccentricities = {}
        
        for u in self.nodes:
            dists = [apsp[u][v] for v in self.nodes if apsp[u][v] != float('inf') and u != v]
            eccentricities[u] = max(dists) if dists else 0.0
            
        ecc_vals = list(eccentricities.values())
        if not ecc_vals:
            return eccentricities, 0.0, 0.0
            
        radius = min(ecc_vals)
        diameter = max(ecc_vals)
        return eccentricities, radius, diameter


    # ==========================================
    # 4. Centrality Measures
    # ==========================================
    
    def degree_centrality(self) -> Dict[any, float]:
        """
        Computes Degree Centrality: fraction of nodes connected to.
        Normalized by N-1.
        """
        n = self.num_nodes
        denom = n - 1 if n > 1 else 1.0
        centrality = {}
        for node in self.nodes:
            deg = self.degree(node)
            # If directed, use out-degree + in-degree or standard degree depending on convention.
            # We will use out-degree + in-degree for directed as standard connectivity centrality.
            deg_val = sum(deg) if isinstance(deg, tuple) else deg
            centrality[node] = deg_val / denom
        return centrality

    def closeness_centrality(self) -> Dict[any, float]:
        """
        Computes Closeness Centrality: reciprocal of the sum of shortest path distances.
        C(u) = (n - 1) / sum(d(u, v) for all v != u).
        If the graph is not fully connected, computes closeness over reachable components (Wasserman-Faust formula).
        """
        n = self.num_nodes
        centrality = {}
        apsp = self.compute_all_pairs_shortest_paths_unweighted()
        
        for u in self.nodes:
            dists = [apsp[u][v] for v in self.nodes if u != v and apsp[u][v] != float('inf')]
            sum_dists = sum(dists)
            reachable = len(dists)
            
            if sum_dists == 0 or reachable == 0:
                centrality[u] = 0.0
            else:
                # Wasserman and Faust formula for disconnected graphs
                raw_closeness = reachable / sum_dists
                norm_factor = reachable / (n - 1)
                centrality[u] = raw_closeness * norm_factor
                
        return centrality

    def betweenness_centrality(self) -> Dict[any, float]:
        """
        Computes Betweenness Centrality using Brandes' Algorithm (O(V*E) time).
        Measures the fraction of shortest paths passing through a node.
        """
        # Initialize betweenness
        betweenness = {node: 0.0 for node in self.nodes}
        
        for s in self.nodes:
            # Single-source shortest paths discovery and DAG building
            # Stack S stores nodes in order of non-decreasing distance from s
            S: List[any] = []
            # P[v] stores list of immediate predecessors of v on shortest paths from s
            P: Dict[any, List[any]] = {w: [] for w in self.nodes}
            # sigma[v] stores number of shortest paths from s to v
            sigma = {w: 0.0 for w in self.nodes}
            sigma[s] = 1.0
            # d[v] stores distance from s to v
            d = {w: -1 for w in self.nodes}
            d[s] = 0
            
            # BFS queue
            queue = [s]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                S.append(v)
                for w in self.adj[v]:
                    # w found for the first time
                    if d[w] < 0:
                        d[w] = d[v] + 1
                        queue.append(w)
                    # shortest path to w via v?
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)
                        
            # Accumulation: Backtrack from leaf nodes to s to compute dependency values
            delta = {w: 0.0 for w in self.nodes}
            while S:
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
                    
        # Normalization
        n = self.num_nodes
        if n <= 2:
            return betweenness
            
        # Normalization factor is 1/((n-1)*(n-2)) for directed, 2/((n-1)*(n-2)) for undirected
        norm = 1.0 / ((n - 1) * (n - 2))
        if not self.directed:
            norm *= 2.0
            
        for node in betweenness:
            betweenness[node] *= norm
            
        return betweenness

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[any, float]:
        """
        Computes Eigenvector Centrality using the Power Iteration method.
        x_k+1 = A * x_k / ||A * x_k||.
        """
        n = self.num_nodes
        if n == 0:
            return {}
            
        # Convert graph to matrix
        adj_matrix, node_order = self.to_adjacency_matrix()
        
        # Initial vector: uniform values
        x = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            x_next = np.dot(adj_matrix, x)
            norm = np.linalg.norm(x_next)
            if norm > 0:
                x_next /= norm
                
            # Convergence check
            if np.linalg.norm(x_next - x) < tol:
                x = x_next
                break
            x = x_next
            
        return {node_order[i]: float(x[i]) for i in range(n)}

    def pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[any, float]:
        """
        Computes PageRank centrality using Power Iteration.
        PR(u) = (1-alpha)/N + alpha * sum(PR(v) / OutDegree(v) for all v -> u)
        
        Args:
            alpha (float): Damping factor (default 0.85).
            max_iter (int): Maximum iteration count.
            tol (float): Convergence tolerance.
        """
        n = self.num_nodes
        if n == 0:
            return {}
            
        node_order = list(self.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_order)}
        
        # Build the transition matrix M: M[i, j] is transition prob from j to i
        # If j is a sink (out_degree = 0), transition probability is uniform 1/N (teleportation)
        M = np.zeros((n, n))
        for j_id, j in enumerate(node_order):
            out_neighbors = list(self.adj[j].keys())
            deg = len(out_neighbors)
            if deg == 0:
                # Sink node: distributes page rank evenly
                M[:, j_id] = 1.0 / n
            else:
                for neighbor in out_neighbors:
                    i_id = node_to_idx[neighbor]
                    M[i_id, j_id] = 1.0 / deg
                    
        # Power Iteration
        # Vector v represents the uniform teleportation vector
        v = np.ones(n) / n
        pr = np.ones(n) / n
        
        for _ in range(max_iter):
            pr_next = alpha * np.dot(M, pr) + (1 - alpha) * v
            # L1 norm check for convergence
            if np.sum(np.abs(pr_next - pr)) < tol:
                pr = pr_next
                break
            pr = pr_next
            
        return {node_order[i]: float(pr[i]) for i in range(n)}


    # ==========================================
    # 5. Spectral Graph Theory
    # ==========================================
    
    def laplacian_matrix(self) -> Tuple[np.ndarray, List[any]]:
        """
        Computes the standard combinatorial Graph Laplacian: L = D - A.
        Returns:
            L (np.ndarray): The Laplacian matrix.
            node_order (list): The node IDs corresponding to matrix rows/cols.
        """
        A, node_order = self.to_adjacency_matrix()
        # Row sum of A gives the out-degree of each node
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        L = D - A
        return L, node_order

    def normalized_laplacian_matrix(self) -> Tuple[np.ndarray, List[any]]:
        """
        Computes the symmetric normalized Graph Laplacian: L_sym = D^(-1/2) * L * D^(-1/2).
        L_sym[i, j] = 1 if i == j and d_i != 0 else -A[i, j]/sqrt(d_i * d_j).
        """
        A, node_order = self.to_adjacency_matrix()
        degrees = np.sum(A, axis=1)
        
        # Calculate D^(-1/2) safely, replacing 0 degree inverses with 0
        with np.errstate(divide='ignore', invalid='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.0
            
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # combinatorial L
        D = np.diag(degrees)
        L = D - A
        
        # symmetric normalization
        L_sym = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
        return L_sym, node_order

    def spectral_decomposition(self, normalized: bool = False) -> Tuple[np.ndarray, np.ndarray, List[any]]:
        """
        Computes the eigenvalues and eigenvectors of the Laplacian matrix.
        Returns:
            eigenvalues (np.ndarray): Sorted eigenvalues in ascending order.
            eigenvectors (np.ndarray): Corresponding eigenvectors (columns).
            node_order (list): Order of nodes.
        """
        if normalized:
            L, node_order = self.normalized_laplacian_matrix()
        else:
            L, node_order = self.laplacian_matrix()
            
        # Solve symmetric eigenvalue problem (hermitian)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # eigh returns sorted values, but let's double check sorted order
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors, node_order

    def spectral_embeddings(self, k: int = 2, normalized: bool = False) -> Tuple[np.ndarray, List[any]]:
        """
        Creates spectral embeddings of nodes using the bottom 'k' non-trivial eigenvectors of the Laplacian.
        Ideal for dimensionality reduction, spectral clustering, and graph drawing (spectral layout).
        
        Args:
            k (int): Number of dimensions (eigenvectors to select).
            normalized (bool): Whether to use the normalized Laplacian.
        """
        eigenvalues, eigenvectors, node_order = self.spectral_decomposition(normalized=normalized)
        
        # The first eigenvalue of combinatorial Laplacian is always 0, corresponding to constant vector.
        # We skip the first eigenvalue (index 0) and take the next 'k' eigenvectors.
        # The second eigenvector is the famous Fiedler Vector, which indicates the sparsest cut!
        if len(eigenvalues) <= k + 1:
            # If not enough nodes, return all eigenvectors except the 1st
            selected_vectors = eigenvectors[:, 1:]
        else:
            selected_vectors = eigenvectors[:, 1:k+1]
            
        return selected_vectors, node_order

    def graph_fourier_transform(self, signal: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Performs the Graph Fourier Transform (GFT) on a spatial node signal.
        Projects the spatial signal onto the eigenvectors (Fourier basis) of the Graph Laplacian.
        
        Args:
            signal (np.ndarray): 1D array of shape (N,) representing node features.
            normalized (bool): True to use normalized Laplacian basis.
            
        Returns:
            spectral_signal (np.ndarray): Signal in the graph spectral domain.
        """
        _, eigenvectors, _ = self.spectral_decomposition(normalized=normalized)
        # GFT: x_hat = U^T * x
        # where U is the matrix of eigenvectors
        return np.dot(eigenvectors.T, signal)

    def inverse_graph_fourier_transform(self, spectral_signal: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Performs the Inverse Graph Fourier Transform (IGFT) to reconstruct the spatial signal.
        Reconstructs spatial features from spectral coefficients: x = U * x_hat.
        """
        _, eigenvectors, _ = self.spectral_decomposition(normalized=normalized)
        return np.dot(eigenvectors, spectral_signal)
