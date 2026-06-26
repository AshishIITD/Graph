"""
Graph Ecosystem Masterclass - Phase 2: Canonical Graph Algorithms & Classical Theory
===================================================================================
This module implements the core graph algorithms frequently asked in technical interviews
and used in optimization pipelines: shortest paths, network flows, connectivity, and ordering.

All implementations are written from scratch and heavily commented.
"""

import heapq
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Set, Union, Optional, Callable
from src.phase1_foundations import Graph

# ==========================================
# 1. Traversals
# ==========================================

def dfs_recursive(graph: Graph, start: any, visited: Optional[Set[any]] = None) -> List[any]:
    """Runs a recursive Depth-First Search (DFS) traversal."""
    if visited is None:
        visited = set()
    visited.add(start)
    path = [start]
    for neighbor in graph.adj[start]:
        if neighbor not in visited:
            path.extend(dfs_recursive(graph, neighbor, visited))
    return path

def dfs_iterative(graph: Graph, start: any) -> List[any]:
    """Runs an iterative Depth-First Search (DFS) traversal using an explicit stack."""
    visited = set()
    stack = [start]
    path = []
    
    while stack:
        curr = stack.pop()
        if curr not in visited:
            visited.add(curr)
            path.append(curr)
            # Add neighbors in reverse order to match recursive behavior
            neighbors = list(graph.adj[curr].keys())
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
    return path

def bfs(graph: Graph, start: any) -> List[any]:
    """Runs a Breadth-First Search (BFS) traversal using a queue."""
    visited = {start}
    queue = deque([start])
    path = []
    
    while queue:
        curr = queue.popleft()
        path.append(curr)
        for neighbor in graph.adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return path


# ==========================================
# 2. Shortest Path Algorithms
# ==========================================

def dijkstra(graph: Graph, start: any) -> Tuple[Dict[any, float], Dict[any, Optional[any]]]:
    """
    Computes single-source shortest paths using Dijkstra's algorithm (Heap-based, O((V+E) log V)).
    Only valid for non-negative edge weights.
    
    Returns:
        distances: dict of node_id -> shortest distance from start.
        predecessors: dict of node_id -> predecessor node in the shortest path.
    """
    distances = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[start] = 0.0
    
    # Priority Queue stores tuples of (distance, node)
    pq = [(0.0, start)]
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        
        # Lazy deletion in heap: skip if we found a shorter path to this node already
        if curr_dist > distances[curr_node]:
            continue
            
        for neighbor, edge_data in graph.adj[curr_node].items():
            weight = edge_data["weight"]
            new_dist = curr_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = curr_node
                heapq.heappush(pq, (new_dist, neighbor))
                
    return distances, predecessors

def bellman_ford(graph: Graph, start: any) -> Tuple[Dict[any, float], Dict[any, Optional[any]], bool]:
    """
    Computes single-source shortest paths using Bellman-Ford algorithm (O(V*E)).
    Supports negative edge weights and detects negative weight cycles.
    
    Returns:
        distances: dict of node_id -> shortest distance.
        predecessors: dict of node_id -> predecessor.
        has_negative_cycle: True if a negative cycle is reachable from start, False otherwise.
    """
    distances = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[start] = 0.0
    
    # Relax edges V-1 times
    nodes = list(graph.nodes.keys())
    edges = graph.to_edge_list()  # (u, v, weight)
    
    for _ in range(len(nodes) - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                
    # Check for negative-weight cycles (the V-th iteration)
    has_negative_cycle = False
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break
            
    return distances, predecessors, has_negative_cycle

def floyd_warshall(graph: Graph) -> Tuple[np.ndarray, List[any]]:
    """
    Computes all-pairs shortest paths using Floyd-Warshall (Dynamic Programming, O(V^3)).
    Works with negative weights, but not negative cycles.
    """
    nodes = list(graph.nodes.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Initialize distance matrix
    dist = np.full((n, n), float('inf'))
    for i in range(n):
        dist[i, i] = 0.0
        
    for u in graph.adj:
        u_idx = node_to_idx[u]
        for v, edge_data in graph.adj[u].items():
            v_idx = node_to_idx[v]
            dist[u_idx, v_idx] = edge_data["weight"]
            
    # Dynamic Programming State Transitions
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    
    return dist, nodes

def a_star(graph: Graph, start: any, goal: any, heuristic_fn: Callable[[any, any], float]) -> Tuple[List[any], float]:
    """
    A* Search Algorithm for pathfinding. Uses a heuristic function h(n) to guide the search.
    f(n) = g(n) + h(n)
    
    Args:
        heuristic_fn: A function taking (node, goal) and returning an estimated cost.
    """
    pq = [(heuristic_fn(start, goal), 0.0, start, [start])] # (f_score, g_score, node, path)
    visited = set()
    
    while pq:
        f_score, g_score, curr, path = heapq.heappop(pq)
        
        if curr == goal:
            return path, g_score
            
        if curr in visited:
            continue
        visited.add(curr)
        
        for neighbor, edge_data in graph.adj[curr].items():
            if neighbor not in visited:
                weight = edge_data["weight"]
                next_g = g_score + weight
                next_f = next_g + heuristic_fn(neighbor, goal)
                heapq.heappush(pq, (next_f, next_g, neighbor, path + [neighbor]))
                
    return [], float('inf')

def johnson(graph: Graph) -> Tuple[Optional[np.ndarray], List[any]]:
    """
    Johnson's Algorithm for All-Pairs Shortest Paths in sparse graphs (O(V*E + V^2 log V)).
    Combines Bellman-Ford (to reweight edges to be non-negative) and Dijkstra.
    """
    nodes = list(graph.nodes.keys())
    n = len(nodes)
    
    # 1. Add a dummy source node 'q' connected to all other nodes with weight 0
    dummy_source = "__johnson_dummy__"
    extended_graph = Graph(directed=True)
    for node in graph.nodes:
        extended_graph.add_node(node)
    for u in graph.adj:
        for v, edge_data in graph.adj[u].items():
            extended_graph.add_edge(u, v, weight=edge_data["weight"])
            
    extended_graph.add_node(dummy_source)
    for node in nodes:
        extended_graph.add_edge(dummy_source, node, weight=0.0)
        
    # 2. Run Bellman-Ford from dummy source 'q' to find node potentials h(u)
    h_dists, _, has_neg_cycle = bellman_ford(extended_graph, dummy_source)
    if has_neg_cycle:
        # Cannot solve: graph contains a negative cycle!
        return None, nodes
        
    # Remove dummy source potentials
    h = {node: h_dists[node] for node in nodes}
    
    # 3. Reweight the edges of the original graph: w'(u, v) = w(u, v) + h(u) - h(v)
    # This guarantees all new weights are non-negative.
    reweighted_graph = Graph(directed=graph.directed)
    for node in graph.nodes:
        reweighted_graph.add_node(node)
    for u in graph.adj:
        for v, edge_data in graph.adj[u].items():
            w_prime = edge_data["weight"] + h[u] - h[v]
            reweighted_graph.add_edge(u, v, weight=w_prime)
            
    # 4. Run Dijkstra for every node on the reweighted graph
    dist_matrix = np.full((n, n), float('inf'))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for u_idx, u in enumerate(nodes):
        dists, _ = dijkstra(reweighted_graph, u)
        for v_idx, v in enumerate(nodes):
            if dists[v] != float('inf'):
                # Undo the reweighting: d(u, v) = d'(u, v) - h(u) + h(v)
                dist_matrix[u_idx, v_idx] = dists[v] - h[u] + h[v]
                
    return dist_matrix, nodes


# ==========================================
# 3. Minimum Spanning Tree (MST)
# ==========================================

class DisjointSet:
    """Disjoint Set (Union-Find) with Path Compression and Union by Rank."""
    def __init__(self, elements: Set[any]):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}
        
    def find(self, x: any) -> any:
        """Finds the representative of the set containing x (with path compression)."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) # Path compression
        return self.parent[x]
        
    def union(self, x: any, y: any) -> bool:
        """Unites the sets containing x and y (with union by rank). Returns True if merged."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
            
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

def kruskal(graph: Graph) -> List[Tuple[any, any, float]]:
    """Kruskal's MST Algorithm (O(E log E)). Sorts edges and merges components via Union-Find."""
    edges = graph.to_edge_list()
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    ds = DisjointSet(set(graph.nodes.keys()))
    mst = []
    
    for u, v, weight in edges:
        if ds.union(u, v):
            mst.append((u, v, weight))
            if len(mst) == graph.num_nodes - 1:
                break
    return mst

def prim(graph: Graph, start_node: Optional[any] = None) -> List[Tuple[any, any, float]]:
    """Prim's MST Algorithm (Heap-based, O(E log V)). Grows the tree from a single node."""
    if not graph.nodes:
        return []
    if start_node is None:
        start_node = list(graph.nodes.keys())[0]
        
    mst = []
    visited = {start_node}
    
    # Heap stores: (weight, u, v) where u is in MST and v is a neighbor to add
    pq = []
    for neighbor, edge_data in graph.adj[start_node].items():
        heapq.heappush(pq, (edge_data["weight"], start_node, neighbor))
        
    while pq and len(visited) < graph.num_nodes:
        weight, u, v = heapq.heappop(pq)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            for next_neighbor, edge_data in graph.adj[v].items():
                if next_neighbor not in visited:
                    heapq.heappush(pq, (edge_data["weight"], v, next_neighbor))
                    
    return mst

def boruvka(graph: Graph) -> List[Tuple[any, any, float]]:
    """
    Boruvka's MST Algorithm (O(E log V)).
    Iteratively finds the cheapest outgoing edge for every component and merges them.
    Fascinating parallel-friendly classical algorithm.
    """
    nodes = list(graph.nodes.keys())
    ds = DisjointSet(set(nodes))
    mst = []
    
    edges = graph.to_edge_list()
    num_components = len(nodes)
    
    while num_components > 1:
        # Track the cheapest edge leaving each component
        # component_root -> (u, v, weight)
        cheapest_edge = {}
        
        for u, v, weight in edges:
            root_u = ds.find(u)
            root_v = ds.find(v)
            if root_u != root_v:
                # Check for component u
                if root_u not in cheapest_edge or weight < cheapest_edge[root_u][2]:
                    cheapest_edge[root_u] = (u, v, weight)
                # Check for component v
                if root_v not in cheapest_edge or weight < cheapest_edge[root_v][2]:
                    cheapest_edge[root_v] = (u, v, weight)
                    
        # Merge components using the cheapest edges found
        merged_any = False
        for root in cheapest_edge:
            u, v, weight = cheapest_edge[root]
            if ds.union(u, v):
                mst.append((u, v, weight))
                num_components -= 1
                merged_any = True
                
        if not merged_any:
            # Graph is not fully connected, return MST of connected components
            break
            
    return mst


# ==========================================
# 4. Connectivity & Ordering
# ==========================================

def tarjan_scc(graph: Graph) -> List[List[any]]:
    """
    Tarjan's strongly connected components algorithm (O(V+E) time, single pass).
    Uses node indices and low-link values on a DFS stack to identify components.
    """
    index = 0
    stack: List[any] = []
    on_stack: Set[any] = set()
    indices: Dict[any, int] = {}
    lowlink: Dict[any, int] = {}
    sccs: List[List[any]] = []
    
    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        
        for w in graph.adj[v]:
            if w not in indices:
                # w has not yet been visited; recurse
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                # w is on the stack and hence in the current SCC
                lowlink[v] = min(lowlink[v], indices[w])
                
        # If v is a root node, pop the stack and generate an SCC
        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for node in graph.nodes:
        if node not in indices:
            strongconnect(node)
            
    return sccs

def kosaraju_scc(graph: Graph) -> List[List[any]]:
    """
    Kosaraju's strongly connected components algorithm (O(V+E) time, double pass).
    1. DFS on original graph to get post-order stack.
    2. DFS on transpose graph in reverse post-order.
    """
    # Step 1: DFS on G to compute finishing order
    visited = set()
    order_stack = []
    
    def dfs_order(v):
        visited.add(v)
        for w in graph.adj[v]:
            if w not in visited:
                dfs_order(w)
        order_stack.append(v)
        
    for node in graph.nodes:
        if node not in visited:
            dfs_order(node)
            
    # Step 2: Build Transpose Graph G^T
    transpose = Graph(directed=True)
    for node in graph.nodes:
        transpose.add_node(node)
    for u in graph.adj:
        for v, edge_data in graph.adj[u].items():
            transpose.add_edge(v, u, weight=edge_data["weight"])
            
    # Step 3: DFS on G^T in order of G's finishing stack (popping from the back)
    visited.clear()
    sccs = []
    
    def dfs_collect(v, component):
        visited.add(v)
        component.append(v)
        for w in transpose.adj[v]:
            if w not in visited:
                dfs_collect(w, component)
                
    while order_stack:
        curr = order_stack.pop()
        if curr not in visited:
            comp = []
            dfs_collect(curr, comp)
            sccs.append(comp)
            
    return sccs

def weakly_connected_components(graph: Graph) -> List[List[any]]:
    """Finds Weakly Connected Components (WCC) in a directed graph by treating it as undirected."""
    undirected = Graph(directed=False)
    for node in graph.nodes:
        undirected.add_node(node)
    for u in graph.adj:
        for v, edge_data in graph.adj[u].items():
            undirected.add_edge(u, v, weight=edge_data["weight"])
            
    # Run simple BFS to collect components
    visited = set()
    wccs = []
    for node in undirected.nodes:
        if node not in visited:
            comp = bfs(undirected, node)
            visited.update(comp)
            wccs.append(comp)
    return wccs

def topological_sort(graph: Graph) -> List[any]:
    """
    Performs Topological Sort of a DAG using Kahn's Algorithm (indegree-based, O(V+E)).
    Throws ValueError if a cycle is detected.
    """
    # Calculate indegrees
    indegrees = {node: 0 for node in graph.nodes}
    for u in graph.adj:
        for v in graph.adj[u]:
            indegrees[v] += 1
            
    # Collect nodes with 0 indegree
    queue = deque([node for node in graph.nodes if indegrees[node] == 0])
    topo_order = []
    
    while queue:
        curr = queue.popleft()
        topo_order.append(curr)
        for neighbor in graph.adj[curr]:
            indegrees[neighbor] -= 1
            if indegrees[neighbor] == 0:
                queue.append(neighbor)
                
    if len(topo_order) != graph.num_nodes:
        raise ValueError("Graph contains a cycle! Topological sort is only defined for DAGs.")
        
    return topo_order


# ==========================================
# 5. Cycle Detection, Coloring & Matching
# ==========================================

def detect_cycle_dfs(graph: Graph) -> bool:
    """
    Detects if a cycle exists in the graph using DFS.
    Supports both directed (using recursion stack tracking) and undirected (ignoring parent edge).
    """
    visited = set()
    
    if graph.directed:
        rec_stack = set()
        def dfs_directed(v) -> bool:
            visited.add(v)
            rec_stack.add(v)
            for neighbor in graph.adj[v]:
                if neighbor not in visited:
                    if dfs_directed(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False
            
        for node in graph.nodes:
            if node not in visited:
                if dfs_directed(node):
                    return True
        return False
    else:
        # Undirected cycle detection
        def dfs_undirected(v, parent) -> bool:
            visited.add(v)
            for neighbor in graph.adj[v]:
                if neighbor not in visited:
                    if dfs_undirected(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False
            
        for node in graph.nodes:
            if node not in visited:
                if dfs_undirected(node, None):
                    return True
        return False

def greedy_coloring(graph: Graph) -> Dict[any, int]:
    """
    Greedy Graph Coloring (vertex coloring).
    Assigns the smallest available color (0, 1, 2...) to each node in arbitrary order.
    """
    colors = {}
    for node in graph.nodes:
        # Find colors used by neighbors
        neighbor_colors = {colors[neighbor] for neighbor in graph.adj[node] if neighbor in colors}
        
        # Find the smallest color not used by neighbors
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
    return colors

def welsh_powell_coloring(graph: Graph) -> Dict[any, int]:
    """
    Welsh-Powell Graph Coloring Algorithm.
    Orders nodes in descending order of their degree, then colors them greedily.
    Often yields fewer colors than standard greedy coloring.
    """
    # Sort nodes by degree in descending order
    # (For directed, degree = indegree + outdegree)
    def get_deg(v):
        deg = graph.degree(v)
        return sum(deg) if isinstance(deg, tuple) else deg
        
    sorted_nodes = sorted(list(graph.nodes.keys()), key=get_deg, reverse=True)
    colors = {}
    
    for node in sorted_nodes:
        neighbor_colors = {colors[neighbor] for neighbor in graph.adj[node] if neighbor in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
        
    return colors

def hopcroft_karp_matching(graph: Graph, left_nodes: Set[any], right_nodes: Set[any]) -> Dict[any, any]:
    """
    Hopcroft-Karp Algorithm for Maximum Cardinality Bipartite Matching (O(E * sqrt(V))).
    Finds augmenting paths using BFS and DFS.
    
    Returns:
        matching: Dict mapping left_node -> right_node and vice-versa.
    """
    # dummy NIL node
    NIL = "__nil__"
    pair_left = {u: NIL for u in left_nodes}
    pair_right = {v: NIL for v in right_nodes}
    dist = {}
    
    def bfs_hk() -> bool:
        queue = deque()
        for u in left_nodes:
            if pair_left[u] == NIL:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        dist[NIL] = float('inf')
        
        while queue:
            u = queue.popleft()
            if dist[u] < dist[NIL]:
                for v in graph.adj[u]:
                    # v is in right_nodes
                    if v in right_nodes:
                        next_u = pair_right[v]
                        if dist[next_u] == float('inf'):
                            dist[next_u] = dist[u] + 1
                            queue.append(next_u)
        return dist[NIL] != float('inf')
        
    def dfs_hk(u) -> bool:
        for v in graph.adj[u]:
            if v in right_nodes:
                next_u = pair_right[v]
                if dist[next_u] == dist[u] + 1:
                    if next_u == NIL or dfs_hk(next_u):
                        pair_left[u] = v
                        pair_right[v] = u
                        return True
        dist[u] = float('inf')
        return False
        
    matching_size = 0
    while bfs_hk():
        for u in left_nodes:
            if pair_left[u] == NIL:
                if dfs_hk(u):
                    matching_size += 1
                    
    # Clean matches (remove NIL entries)
    matching = {u: v for u, v in pair_left.items() if v != NIL}
    # Also add right to left mappings
    for v, u in pair_right.items():
        if u != NIL:
            matching[v] = u
            
    return matching


# ==========================================
# 6. Network Flow (Max-Flow & Min-Cut)
# ==========================================

class FlowNetwork:
    """Helper structure representing a residual graph for flow calculations."""
    def __init__(self, graph: Graph):
        self.nodes = list(graph.nodes.keys())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Residual capacity matrix
        self.capacity = np.zeros((self.n, self.n))
        for u in graph.adj:
            u_idx = self.node_to_idx[u]
            for v, edge_data in graph.adj[u].items():
                v_idx = self.node_to_idx[v]
                self.capacity[u_idx, v_idx] = edge_data["weight"]

def edmonds_karp(graph: Graph, source: any, sink: any) -> Tuple[float, Dict[Tuple[any, any], float], List[Tuple[any, any]]]:
    """
    Edmonds-Karp algorithm for Max-Flow / Min-Cut (BFS augmenting paths, O(V * E^2)).
    
    Returns:
        max_flow: The maximum flow value.
        flow_dict: Map of (u, v) -> actual flow.
        min_cut: List of edges (u, v) crossing the minimum cut boundary.
    """
    fn = FlowNetwork(graph)
    s = fn.node_to_idx[source]
    t = fn.node_to_idx[sink]
    
    max_flow = 0.0
    
    def bfs_augment() -> Tuple[bool, List[int]]:
        parent = [-1] * fn.n
        parent[s] = s
        queue = deque([s])
        
        while queue:
            curr = queue.popleft()
            if curr == t:
                return True, parent
            for neighbor in range(fn.n):
                if parent[neighbor] == -1 and fn.capacity[curr, neighbor] > 0:
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return False, parent

    # Augmentation loops
    while True:
        has_path, parent = bfs_augment()
        if not has_path:
            break
            
        # Find bottleneck capacity along the path
        bottleneck = float('inf')
        curr = t
        while curr != s:
            p = parent[curr]
            bottleneck = min(bottleneck, fn.capacity[p, curr])
            curr = p
            
        # Update residual capacities
        curr = t
        while curr != s:
            p = parent[curr]
            fn.capacity[p, curr] -= bottleneck
            fn.capacity[curr, p] += bottleneck
            curr = p
            
        max_flow += bottleneck
        
    # Reconstruct flows: flow(u, v) = original_capacity(u, v) - residual_capacity(u, v)
    flow_dict = {}
    original_capacities = FlowNetwork(graph)
    for u in graph.adj:
        u_idx = fn.node_to_idx[u]
        for v in graph.adj[u]:
            v_idx = fn.node_to_idx[v]
            actual_flow = max(0.0, original_capacities.capacity[u_idx, v_idx] - fn.capacity[u_idx, v_idx])
            flow_dict[(u, v)] = actual_flow
            
    # Extract Min-Cut: find all nodes reachable from source in the final residual graph
    visited = [False] * fn.n
    visited[s] = True
    queue = deque([s])
    while queue:
        curr = queue.popleft()
        for neighbor in range(fn.n):
            if not visited[neighbor] and fn.capacity[curr, neighbor] > 0:
                visited[neighbor] = True
                queue.append(neighbor)
                
    min_cut_edges = []
    for u in graph.adj:
        u_idx = fn.node_to_idx[u]
        for v in graph.adj[u]:
            v_idx = fn.node_to_idx[v]
            # If u is reachable and v is NOT reachable, this edge is part of the min-cut!
            if visited[u_idx] and not visited[v_idx]:
                min_cut_edges.append((u, v))
                
    return max_flow, flow_dict, min_cut_edges

def dinic(graph: Graph, source: any, sink: any) -> Tuple[float, Dict[Tuple[any, any], float]]:
    """
    Dinic's Algorithm for Max Flow (O(V^2 * E)).
    Uses BFS to construct a level graph and DFS to find blocking flows.
    Significantly faster than Edmonds-Karp for dense networks.
    """
    fn = FlowNetwork(graph)
    s = fn.node_to_idx[source]
    t = fn.node_to_idx[sink]
    
    level = [-1] * fn.n
    
    def bfs_level_graph() -> bool:
        """Constructs the level graph and returns True if sink is reachable."""
        for i in range(fn.n):
            level[i] = -1
        level[s] = 0
        queue = deque([s])
        
        while queue:
            curr = queue.popleft()
            for neighbor in range(fn.n):
                if level[neighbor] < 0 and fn.capacity[curr, neighbor] > 0:
                    level[neighbor] = level[curr] + 1
                    queue.append(neighbor)
        return level[t] >= 0
        
    def dfs_blocking_flow(curr: int, push: float, ptr: List[int]) -> float:
        """DFS to find and push blocking flows along level-graph edges."""
        if push == 0.0 or curr == t:
            return push
            
        # ptr (pointer array) prevents re-exploring dead-end edges
        while ptr[curr] < fn.n:
            neighbor = ptr[curr]
            if level[neighbor] == level[curr] + 1 and fn.capacity[curr, neighbor] > 0:
                # Can we push flow?
                next_push = min(push, fn.capacity[curr, neighbor])
                pushed = dfs_blocking_flow(neighbor, next_push, ptr)
                if pushed > 0:
                    fn.capacity[curr, neighbor] -= pushed
                    fn.capacity[neighbor, curr] += pushed
                    return pushed
            ptr[curr] += 1
        return 0.0
        
    max_flow = 0.0
    while bfs_level_graph():
        # ptr stores the index of the first edge that can be explored from node i
        ptr = [0] * fn.n
        while True:
            pushed = dfs_blocking_flow(s, float('inf'), ptr)
            if pushed == 0.0:
                break
            max_flow += pushed
            
    # Reconstruct flows
    flow_dict = {}
    original_capacities = FlowNetwork(graph)
    for u in graph.adj:
        u_idx = fn.node_to_idx[u]
        for v in graph.adj[u]:
            v_idx = fn.node_to_idx[v]
            actual_flow = max(0.0, original_capacities.capacity[u_idx, v_idx] - fn.capacity[u_idx, v_idx])
            flow_dict[(u, v)] = actual_flow
            
    return max_flow, flow_dict
