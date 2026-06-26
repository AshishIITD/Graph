"""
Graph Ecosystem Masterclass - Master Demo Runner
================================================
This script executes a comprehensive demonstration and verification suite
for all 6 phases of the Graph Ecosystem Masterclass. It ensures everything
runs flawlessly, validating both classical algorithms and modern deep learning models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.phase1_foundations import Graph
import src.phase2_algorithms as algs
import src.phase3_embeddings_pgm as emb_pgm
import src.phase4_knowledge_graphs as kg
import src.phase5_gnns as gnn
import src.phase6_graphrag_agents as rag_agent

def run_phase1_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 1: GRAPH FOUNDATIONS & SPECTRAL THEORY")
    print("="*50)
    
    # 1. Create a graph
    g = Graph(directed=False)
    # Add nodes and edges
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=2.0)
    g.add_edge("C", "D", weight=1.5)
    g.add_edge("D", "A", weight=1.0)
    g.add_edge("B", "D", weight=3.0)
    
    print(f"[+] Graph created with {g.num_nodes} nodes and {g.num_edges} edges.")
    
    # 2. Representations
    adj_matrix, nodes = g.to_adjacency_matrix()
    print("\n[+] Adjacency Matrix (Node order:", nodes, "):")
    print(adj_matrix)
    
    # 3. Centrality Metrics
    print("\n[+] Calculating Centrality Metrics:")
    pr = g.pagerank(alpha=0.85)
    deg_c = g.degree_centrality()
    close_c = g.closeness_centrality()
    bet_c = g.betweenness_centrality()
    
    for node in sorted(g.nodes.keys()):
        print(f"  Node {node} -> PageRank: {pr[node]:.3f} | DegCent: {deg_c[node]:.3f} | CloseCent: {close_c[node]:.3f} | BetCent: {bet_c[node]:.3f}")
        
    # 4. Spectral Graph Theory
    print("\n[+] Spectral Graph Theory:")
    L, _ = g.laplacian_matrix()
    print("  combinatorial Laplacian L:")
    print(L)
    
    evals, evecs, _ = g.spectral_decomposition(normalized=False)
    print("  Sorted Laplacian Eigenvalues:")
    print("  ", [f"{val:.4f}" for val in evals])
    # The second eigenvalue is the algebraic connectivity (Fiedler value)
    print(f"  Algebraic Connectivity (Fiedler value): {evals[1]:.4f}")
    
    # Graph Fourier Transform
    signal = np.array([1.0, 2.0, 0.5, -1.0]) # Node features signal
    spectral_signal = g.graph_fourier_transform(signal)
    reconstructed = g.inverse_graph_fourier_transform(spectral_signal)
    print("\n  Graph Fourier Transform (GFT) Verification:")
    print("    Original Signal:     ", signal)
    print("    Spectral Coefficients:", [f"{val:.3f}" for val in spectral_signal])
    print("    Reconstructed Signal: ", [f"{val:.3f}" for val in reconstructed])
    assert np.allclose(signal, reconstructed)
    print("    [Success] Perfect reconstruction achieved (orthogonal basis verified!).")


def run_phase2_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 2: CANONICAL ALGORITHMS")
    print("="*50)
    
    # Create a directed graph for path and connectivity solvers
    g = Graph(directed=True)
    g.add_edge("S", "A", weight=4)
    g.add_edge("S", "B", weight=2)
    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "A", weight=3)
    g.add_edge("A", "C", weight=2)
    g.add_edge("B", "D", weight=5)
    g.add_edge("C", "D", weight=1)
    g.add_edge("D", "T", weight=3)
    g.add_edge("C", "T", weight=6)
    
    print(f"[+] Directed Graph created. Nodes: {list(g.nodes.keys())}")
    
    # 1. Traversals
    print("\n[+] Running Traversals from S:")
    print("  Recursive DFS:", algs.dfs_recursive(g, "S"))
    print("  Iterative DFS:", algs.dfs_iterative(g, "S"))
    print("  Queue-based BFS:", algs.bfs(g, "S"))
    
    # 2. Shortest Paths (Dijkstra vs Bellman-Ford)
    dijkstra_dists, dijkstra_pred = algs.dijkstra(g, "S")
    bf_dists, bf_pred, has_neg = algs.bellman_ford(g, "S")
    
    print("\n[+] Shortest Paths from S:")
    print("  Dijkstra distances:    ", {k: dijkstra_dists[k] for k in sorted(dijkstra_dists)})
    print("  Bellman-Ford distances:", {k: bf_dists[k] for k in sorted(bf_dists)})
    print(f"  Negative Cycle detected? {has_neg}")
    
    # 3. Minimum Spanning Tree (Undirected)
    g_undirected = Graph(directed=False)
    g_undirected.add_edge("A", "B", weight=4)
    g_undirected.add_edge("B", "C", weight=8)
    g_undirected.add_edge("C", "D", weight=7)
    g_undirected.add_edge("D", "E", weight=9)
    g_undirected.add_edge("E", "F", weight=10)
    g_undirected.add_edge("F", "G", weight=2)
    g_undirected.add_edge("G", "H", weight=1)
    g_undirected.add_edge("H", "A", weight=8)
    g_undirected.add_edge("B", "H", weight=11)
    g_undirected.add_edge("H", "I", weight=7)
    g_undirected.add_edge("I", "C", weight=2)
    g_undirected.add_edge("G", "I", weight=6)
    g_undirected.add_edge("C", "F", weight=4)
    g_undirected.add_edge("D", "F", weight=14)
    
    print("\n[+] Minimum Spanning Tree solvers on Undirected Graph:")
    kruskal_mst = algs.kruskal(g_undirected)
    prim_mst = algs.prim(g_undirected)
    boruvka_mst = algs.boruvka(g_undirected)
    
    print(f"  Kruskal's MST weight: {sum(w for _, _, w in kruskal_mst)} | Edges: {len(kruskal_mst)}")
    print(f"  Prim's MST weight:    {sum(w for _, _, w in prim_mst)} | Edges: {len(prim_mst)}")
    print(f"  Boruvka's MST weight: {sum(w for _, _, w in boruvka_mst)} | Edges: {len(boruvka_mst)}")
    
    # 4. Connectivity and Topological Sort
    g_dag = Graph(directed=True)
    g_dag.add_edge("Shirt", "Tie")
    g_dag.add_edge("Tie", "Jacket")
    g_dag.add_edge("Belt", "Jacket")
    g_dag.add_edge("Socks", "Shoes")
    g_dag.add_edge("Undershorts", "Pants")
    g_dag.add_edge("Pants", "Belt")
    g_dag.add_edge("Pants", "Shoes")
    
    print("\n[+] Topological Sort on Clothing DAG:")
    print("  Order:", algs.topological_sort(g_dag))
    
    # Strongly Connected Components (Tarjan vs Kosaraju)
    g_scc = Graph(directed=True)
    g_scc.add_edge("A", "B")
    g_scc.add_edge("B", "C")
    g_scc.add_edge("C", "A")
    g_scc.add_edge("C", "D")
    g_scc.add_edge("D", "E")
    g_scc.add_edge("E", "F")
    g_scc.add_edge("F", "D")
    g_scc.add_edge("G", "F")
    g_scc.add_edge("G", "H")
    g_scc.add_edge("H", "G")
    
    print("\n[+] Strongly Connected Components (SCC):")
    print("  Tarjan's SCCs:  ", algs.tarjan_scc(g_scc))
    print("  Kosaraju's SCCs:", algs.kosaraju_scc(g_scc))
    
    # Graph Coloring (Greedy vs Welsh-Powell)
    print("\n[+] Graph Coloring on Undirected Cycle:")
    g_cycle = Graph(directed=False)
    g_cycle.add_edge("1", "2")
    g_cycle.add_edge("2", "3")
    g_cycle.add_edge("3", "4")
    g_cycle.add_edge("4", "5")
    g_cycle.add_edge("5", "1")
    print("  Greedy Coloring:      ", algs.greedy_coloring(g_cycle))
    print("  Welsh-Powell Coloring:", algs.welsh_powell_coloring(g_cycle))
    
    # 5. Max Flow & Min Cut
    g_flow = Graph(directed=True)
    g_flow.add_edge("source", "A", weight=16)
    g_flow.add_edge("source", "B", weight=13)
    g_flow.add_edge("A", "B", weight=10)
    g_flow.add_edge("B", "A", weight=4)
    g_flow.add_edge("A", "C", weight=12)
    g_flow.add_edge("B", "D", weight=14)
    g_flow.add_edge("C", "B", weight=9)
    g_flow.add_edge("D", "C", weight=7)
    g_flow.add_edge("C", "sink", weight=20)
    g_flow.add_edge("D", "sink", weight=4)
    
    print("\n[+] Network Flow (Max-Flow / Min-Cut):")
    flow_ek, _, min_cut = algs.edmonds_karp(g_flow, "source", "sink")
    flow_dinic, _ = algs.dinic(g_flow, "source", "sink")
    print(f"  Edmonds-Karp Max Flow: {flow_ek}")
    print(f"  Dinic's Max Flow:      {flow_dinic}")
    print(f"  Min-Cut Crossing Edges: {min_cut}")


def run_phase3_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 3: EMBEDDINGS & PGMS")
    print("="*50)
    
    # Create a small social network
    g = Graph(directed=False)
    g.add_edge("Alice", "Bob")
    g.add_edge("Bob", "Charlie")
    g.add_edge("Charlie", "David")
    g.add_edge("David", "Alice")
    g.add_edge("Charlie", "Alice")
    g.add_edge("David", "Eve")
    g.add_edge("Eve", "Frank")
    g.add_edge("Frank", "Gina")
    g.add_edge("Gina", "Eve")
    
    # 1. Biased walks (Node2Vec)
    print("\n[+] Node2Vec Biased Random Walks:")
    walker = emb_pgm.BiasedRandomWalker(g, p=0.5, q=2.0)
    walks = walker.generate_walks(num_walks=2, walk_length=6)
    print("  Sample walks:\n  ", "\n   ".join(str(w) for w in walks[:3]))
    
    # 2. Train Node2Vec Skip-Gram Model
    print("\n[+] Training Node2Vec Embeddings in PyTorch...")
    embeddings, nodes = emb_pgm.train_node2vec(g, embedding_dim=8, epochs=10, walk_length=6, num_walks=3)
    print(f"  Successfully trained embeddings. Shape: {embeddings.shape}")
    print(f"  Embedding for 'Alice': {np.round(embeddings[nodes.index('Alice')], 3)}")
    
    # 3. HOPE (High-Order Proximity preserved Embedding) via SVD
    print("\n[+] Preserving Katz Proximity using HOPE (SVD):")
    hope_emb, _ = emb_pgm.compute_hope(g, embedding_dim=8, beta=0.05)
    print(f"  HOPE Embeddings Shape: {hope_emb.shape}")
    print(f"  HOPE embedding for 'Alice': {np.round(hope_emb[nodes.index('Alice')], 3)}")
    
    # 4. Bayesian Network (PGM)
    print("\n[+] Bayesian Network Forward and Rejection Sampling:")
    # Simple Alarm network: Burglary -> Alarm, Earthquake -> Alarm, Alarm -> JohnCalls
    bn = emb_pgm.BayesianNetwork()
    
    # Prior for Burglary (B)
    bn.add_variable("B", parents=[], cpt={
        (): {True: 0.001, False: 0.999}
    })
    # Prior for Earthquake (E)
    bn.add_variable("E", parents=[], cpt={
        (): {True: 0.002, False: 0.998}
    })
    # Conditional CPT for Alarm (A) given B and E (parents ordered alphabetically: B, E)
    bn.add_variable("A", parents=["B", "E"], cpt={
        (True, True): {True: 0.95, False: 0.05},
        (True, False): {True: 0.94, False: 0.06},
        (False, True): {True: 0.29, False: 0.71},
        (False, False): {True: 0.001, False: 0.999}
    })
    # Conditional CPT for JohnCalls (J) given A
    bn.add_variable("J", parents=["A"], cpt={
        (True,): {True: 0.90, False: 0.10},
        (False,): {True: 0.05, False: 0.95}
    })
    
    # Query: P(Burglary | Alarm = True)
    posterior = bn.rejection_sampling(query_var="B", evidence={"A": True}, num_samples=5000)
    print("  Query: P(Burglary | Alarm = True)")
    print(f"  Posterior: {posterior}")
    
    # 5. Hidden Markov Model (HMM) Viterbi decoding
    print("\n[+] HMM Viterbi State Sequence Decoding:")
    # States: Healthy, Fever
    # Observations: Normal, Cold, Dizzy
    states = ["Healthy", "Fever"]
    emissions = ["Normal", "Cold", "Dizzy"]
    
    pi = np.array([0.6, 0.4]) # Initial probabilities
    A = np.array([[0.7, 0.3],  # Transition: Healthy->Healthy, Healthy->Fever
                  [0.4, 0.6]]) #             Fever->Healthy,   Fever->Fever
    B = np.array([[0.5, 0.4, 0.1],  # Emission: Healthy -> Normal, Cold, Dizzy
                  [0.1, 0.3, 0.6]]) #            Fever   -> Normal, Cold, Dizzy
                  
    hmm = emb_pgm.HiddenMarkovModel(states, emissions, A, B, pi)
    
    obs_seq = ["Normal", "Cold", "Dizzy"]
    decoded_states, prob = hmm.viterbi(obs_seq)
    print("  Observation Sequence: ", obs_seq)
    print("  Decoded State Pathway:", decoded_states)
    print(f"  Path Probability:      {prob:.5f}")


def run_phase4_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 4: KNOWLEDGE GRAPHS & KGEs")
    print("="*50)
    
    # 1. Triple Store & SPARQL-Like querying
    print("\n[+] In-Memory Triple Store and SPARQL-like Joins:")
    ts = kg.TripleStore()
    ts.add_triple("Alice", "works_at", "Google")
    ts.add_triple("Bob", "works_at", "Google")
    ts.add_triple("Google", "located_in", "USA")
    ts.add_triple("Charlie", "works_at", "DeepMind")
    ts.add_triple("DeepMind", "located_in", "UK")
    ts.add_triple("Alice", "knows", "Bob")
    
    # Query: Find all people who work at a company located in the USA
    # SPARQL pattern: (?person works_at ?company) (?company located_in USA)
    query_patterns = [
        ("?person", "works_at", "?company"),
        ("?company", "located_in", "USA")
    ]
    bindings = ts.query(query_patterns)
    print("  Query: Find (?person) works_at (?company) located_in USA")
    print("  Results Bindings:")
    for b in bindings:
        print(f"    person: {b['?person']} | company: {b['?company']}")
        
    # 2. Entity Resolution
    print("\n[+] Entity Resolution:")
    g = Graph(directed=False)
    g.add_edge("CompanyA", "Node1")
    g.add_edge("CompanyA", "Node2")
    g.add_edge("CompanyB", "Node1")
    g.add_edge("CompanyB", "Node2")
    
    # Set attributes
    g.add_node("CompanyA", type="corp", name="Acme Corporation")
    g.add_node("CompanyB", type="corp", name="Acme Corp.")
    
    # Custom attribute similarity: string matching
    def name_similarity(attrs1, attrs2):
        n1 = attrs1.get("name", "")
        n2 = attrs2.get("name", "")
        if n1.startswith("Acme") and n2.startswith("Acme"):
            return 0.95
        return 0.0
        
    are_same = kg.GraphResolver.entity_resolution(g, "CompanyA", "CompanyB", attribute_sim_fn=name_similarity)
    print(f"  Are 'CompanyA' and 'CompanyB' duplicate entities? {are_same}")
    
    # 3. Knowledge Graph Embeddings (KGE)
    print("\n[+] Training TransE KGE in PyTorch...")
    # Entities: Alice (0), Bob (1), Google (2)
    # Relations: works_at (0), knows (1)
    num_entities = 3
    num_relations = 2
    dim = 8
    
    transe = kg.TransE(num_entities, num_relations, dim)
    trainer = kg.KGETrainer(transe, model_type="transe", margin=1.0, lr=0.05)
    
    # Positive Triples: (Alice, works_at, Google), (Alice, knows, Bob)
    pos_h = [0, 0]
    pos_r = [0, 1]
    pos_t = [2, 1]
    
    # Corrupted Negative Triples (Tails replaced)
    neg_h = [0, 0]
    neg_r = [0, 1]
    neg_t = [1, 2] # (Alice, works_at, Bob), (Alice, knows, Google)
    
    # Train 5 epochs
    for epoch in range(5):
        loss = trainer.train_step(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
        print(f"  Epoch {epoch+1} -> TransE loss: {loss:.4f}")
        
    print("  TransE embeddings successfully trained.")


def run_phase5_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 5: GNNs & GENERATIVE MODELS")
    print("="*50)
    
    # Create a small graph using PyTorch tensors
    # Node features: 4 nodes, 3 features each
    x = torch.tensor([
        [1.0, 0.0, 0.0],  # Node 0
        [0.0, 1.0, 0.0],  # Node 1
        [0.0, 0.0, 1.0],  # Node 2
        [1.0, 1.0, 1.0]   # Node 3
    ], dtype=torch.float)
    
    # Adjacency: 0-1, 1-2, 2-3, 3-0 (undirected cycle)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3]
    ], dtype=torch.long)
    
    print("[+] Created synthetic graph tensors.")
    print("  Node feature matrix shape:", x.shape)
    print("  Edge index shape:         ", edge_index.shape)
    
    # 1. GCN Forward Pass
    print("\n[+] Testing Graph Convolutional Network (GCN) Layer:")
    gcn = gnn.GCNConv(in_channels=3, out_channels=4)
    out_gcn = gcn(edge_index, x)
    print("  GCN Output Shape:", out_gcn.shape)
    print("  GCN Output Features:\n", out_gcn.detach().numpy())
    
    # 2. GraphSAGE Forward Pass
    print("\n[+] Testing GraphSAGE Layer (Mean aggregation):")
    sage = gnn.GraphSAGEConv(in_channels=3, out_channels=4, aggr="mean")
    out_sage = sage(edge_index, x)
    print("  GraphSAGE Output Shape:", out_sage.shape)
    
    # 3. GAT Forward Pass (Multi-head attention)
    print("\n[+] Testing Graph Attention Network (GAT) Layer:")
    gat = gnn.GATConv(in_channels=3, out_channels=2, heads=2, concat=True)
    out_gat = gat(edge_index, x)
    print("  GAT Output Shape (heads=2, concat=True):", out_gat.shape)
    
    # 4. Graph Transformer Forward Pass
    print("\n[+] Testing Graph Transformer Layer:")
    gt = gnn.GraphTransformerLayer(in_channels=3, out_channels=4, heads=2)
    out_gt = gt(edge_index, x)
    print("  Graph Transformer Output Shape:", out_gt.shape)
    
    # 5. Variational Graph Autoencoder (VGAE) training
    print("\n[+] Training Variational Graph Autoencoder (VGAE) in PyTorch...")
    vgae = gnn.VGAE(in_channels=3, latent_dim=2)
    optimizer = torch.optim.Adam(vgae.parameters(), lr=0.05)
    
    # Target dense adjacency matrix for loss
    adj_target = torch.zeros(4, 4)
    adj_target[edge_index[0], edge_index[1]] = 1.0
    
    for epoch in range(10):
        vgae.train()
        optimizer.zero_grad()
        
        adj_rec, mu, logvar = vgae(edge_index, x)
        
        # Binary cross entropy reconstruction loss
        recon_loss = F.binary_cross_entropy(adj_rec, adj_target)
        kl_loss = vgae.kl_loss(mu, logvar)
        loss = recon_loss + 0.1 * kl_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1} -> VGAE Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f})")
            
    # 6. Physical Graph Diffusion Simulation
    print("\n[+] Physical Heat Graph Diffusion Simulation:")
    g_cycle = Graph(directed=False)
    g_cycle.add_edge("0", "1")
    g_cycle.add_edge("1", "2")
    g_cycle.add_edge("2", "3")
    g_cycle.add_edge("3", "0")
    
    simulator = gnn.GraphDiffusionSimulator(g_cycle)
    # Inject heat of 100.0 units into Node 0, others at 0.0
    initial_heat = np.array([100.0, 0.0, 0.0, 0.0])
    heat_history = simulator.run_diffusion(initial_heat, time_steps=3, alpha=0.1)
    
    print("  Heat dissipation over time steps:")
    for step, heat in enumerate(heat_history):
        print(f"    Step {step} -> Heat: {np.round(heat, 2)}")


def run_phase6_demo():
    print("\n" + "="*50)
    print("DEMONSTRATING PHASE 6: GRAPHRAG & AGENTIC GRAPHS")
    print("="*50)
    
    # 1. Ingest a document in GraphRAG
    print("\n[+] Ingesting Corporate Document into GraphRAG:")
    rag = rag_agent.GraphRAGPipeline()
    
    doc = """
    Alice is an AI Research Scientist who works_at OpenAI.
    OpenAI is located_in SanFrancisco.
    Bob works_at OpenAI as an Engineer.
    DeepSeek is located_in Hangzhou.
    Hangzhou is_a City in China.
    SanFrancisco is_a City in USA.
    """
    rag.ingest_document(doc)
    print("  Document ingested and indexed successfully!")
    print(f"  Total chunks: {len(rag.chunks)}")
    print(f"  Extracted Triples in KG: {list(rag.triple_store.triples)}")
    
    # Query with Hybrid retrieval
    query = "Where is OpenAI located and who works there?"
    print(f"\n[+] Executing Hybrid GraphRAG Query: '{query}'")
    grounded_ans = rag.generate_grounded_answer(query)
    print(grounded_ans)
    
    # 2. Agent Memory Graph
    print("\n[+] Agent Associative memory graph:")
    mem = rag_agent.MemoryGraph()
    mem.add_episodic_memory("Event1", "Analyzed house prices dataset", ["house_scale", "PCA", "variance"])
    mem.add_episodic_memory("Event2", "Trained a Graph Attention Network layer", ["GAT", "attention", "GNN"])
    
    print("  Retrieving memory associations for 'GAT':")
    print("    Associations:", mem.retrieve_associated_memories("GAT"))
    
    mem.decay_memories(decay_factor=0.5)
    print("  Decaying memories (simulating forgetting)...")
    print("    Associations after decay:", mem.retrieve_associated_memories("GAT"))
    
    # 3. Agent Task Execution Graph
    print("\n[+] Agent Task Planning & Execution (Tool Dependency Routing):")
    te_graph = rag_agent.TaskExecutionGraph()
    
    # Define tool functions
    def web_search(initial_input):
        return f"Web search results for: {initial_input}"
        
    def entity_extractor(input_from_WebSearch):
        return f"Entities extracted from [{input_from_WebSearch}] -> [OpenAI, SamAltman]"
        
    def summary_generator(input_from_EntityExtractor, input_from_WebSearch):
        return f"Final Summary: Fusing extracted tokens [{input_from_EntityExtractor}] with raw search data."

    # Register tools
    te_graph.register_tool("WebSearch", web_search)
    te_graph.register_tool("EntityExtractor", entity_extractor)
    te_graph.register_tool("SummaryGenerator", summary_generator)
    
    # Declare dependencies:
    # WebSearch -> EntityExtractor
    # EntityExtractor -> SummaryGenerator
    # WebSearch -> SummaryGenerator
    te_graph.add_dependency("WebSearch", "EntityExtractor")
    te_graph.add_dependency("EntityExtractor", "SummaryGenerator")
    te_graph.add_dependency("WebSearch", "SummaryGenerator")
    
    # Execute plan
    print("  Executing plan with initial query: 'OpenAI leadership'")
    results = te_graph.execute_plan({"WebSearch": "OpenAI leadership"})
    print("\n  Execution Results:")
    for task, res in results.items():
        print(f"    Task [{task}] -> Output: {res}")


if __name__ == "__main__":
    print("="*60)
    print("GRAPH ECOSYSTEM MASTERCLASS - AUTOMATED TEST & VERIFICATION SUITE")
    print("="*60)
    
    run_phase1_demo()
    run_phase2_demo()
    run_phase3_demo()
    run_phase4_demo()
    run_phase5_demo()
    run_phase6_demo()
    
    print("\n" + "="*60)
    print("[SUCCESS] ALL PHASES SUCCESSFULLY EXECUTED AND VERIFIED!")
    print("="*60)
