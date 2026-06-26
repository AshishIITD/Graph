# 🌐 Graph Ecosystem Masterclass (2026+)

Welcome to the **Graph Ecosystem Masterclass**—a complete, production-grade, from-scratch pedagogical and technical reference for advanced AI engineering. 

This repository implements the entire graph spectrum—from classical graph foundations and spectral theory to PyTorch deep Graph Neural Networks (GNNs), Knowledge Graph Embeddings (KGEs), and state-of-the-art enterprise **GraphRAG** and agentic task-planning graphs. All implementations are written in **pure Python, NumPy, SciPy, and PyTorch**, with zero heavy external graph libraries (like `PyG` or `DGL`) to guarantee ultimate pedagogical transparency and mathematical clarity.

---

## 🗺️ Curriculum Overview

The masterclass is divided into **6 learning phases**, each represented by a modular Python file in the package:

### 1. 📐 Phase 1: Graph Foundations & Spectral Theory (`src/phase1_foundations.py`)
*   **Representations**: Conversions between Adjacency Matrix, Adjacency List, Edge List, Incidence Matrix, and Sparse Formats.
*   **Centrality Metrics**: Degree Centrality, Closeness Centrality, PageRank power iterations, and Brandes' O(VE) Betweenness Centrality.
*   **Spectral Graph Theory**: Combinatorial Laplacian ($L = D - A$), Algebraic Connectivity (Fiedler Value), Fiedler Vector bi-partitioning, and the orthogonal Graph Fourier Transform (GFT) with perfect reconstruction.

### 🛣️ Phase 2: Canonical Graph Algorithms (`src/phase2_algorithms.py`)
*   **Traversals**: Recursive/Iterative DFS, Queue-based BFS, and Union-Find (Disjoint Set Union) cycle detection.
*   **Shortest Paths**: Dijkstra's algorithm, Bellman-Ford (with negative cycle detection), Floyd-Warshall, A*, and Johnson's all-pairs algorithm.
*   **Spanning Trees**: Kruskal's, Prim's, and Boruvka's Minimum Spanning Tree (MST) algorithms.
*   **Connectivity & Topology**: Tarjan's and Kosaraju's Strongly Connected Components (SCC), Topological Sort (Kahn's), Welsh-Powell and Greedy Graph Coloring, and Hopcroft-Karp Bipartite Matching.
*   **Network Flows**: Edmonds-Karp and Dinic's Max-Flow algorithms, and the Max-Flow/Min-Cut boundary crossing partition.

### 🧠 Phase 3: Graph Embeddings & Probabilistic Graphical Models (`src/phase3_embeddings_pgm.py`)
*   **Embeddings**: Node2Vec biased random walk generators (DFS/BFS trade-off via $p$ and $q$ parameters), Skip-Gram PyTorch model training for node embeddings, and HOPE (High-Order Proximity preserved Embeddings) using SVD.
*   **PGMs**: Bayesian Network DAG structures with rejection sampling, Markov Random Fields (MRFs), Factor Graphs with sum-product Belief Propagation, Hidden Markov Models (HMM) Viterbi decoding, and Linear-Chain Conditional Random Fields (CRFs).

### 🏷️ Phase 4: Knowledge Graphs & KGE Models (`src/phase4_knowledge_graphs.py`)
*   **KG Engine**: In-memory `TripleStore` with SPO, POS, and OSP index registries supporting O(1) multi-hop SPARQL-like pattern matching joins.
*   **Entity Linking & Resolution**: Rule-based and Jaro-Winkler string similarity systems.
*   **Knowledge Graph Embeddings (KGE)**: Seven major KGE translation and bilinear architectures implemented in PyTorch: `TransE`, `TransH`, `TransR`, `RotatE`, `DistMult`, `ComplEx`, and `RESCAL`.

### 🧬 Phase 5: Graph Neural Networks & Generative Models (`src/phase5_gnns.py`)
*   **MessagePassing Base**: A custom PyTorch message-passing template decoupling `message`, `aggregate` (sum, mean, max via a generalized, dimension-agnostic `scatter_add` implementation), and `update`.
*   **GNN Architectures**: Graph Convolutional Networks (`GCNConv`), `GraphSAGEConv` (mean/max pooling), Graph Attention Networks (`GATConv` with multi-head self-attention), and `GraphTransformerLayer` (with dynamic residual projection shortcuts).
*   **Advanced Models**: Relational GCN (`RGCNConv`) for heterogeneous graphs, Temporal Graph Networks (`TemporalMemoryGNN` with GRU memory cells), and self-supervised Deep Graph Infomax (`DGI`).
*   **Generative Systems**: Variational Graph Autoencoders (`VGAE`) optimized via the Evidence Lower Bound (ELBO), and physical discrete-time Heat Graph Diffusion.

### 🤖 Phase 6: Enterprise GraphRAG & Agentic Graphs (`src/phase6_graphrag_agents.py`)
*   **Vector Store**: A from-scratch TF-IDF Vector Database with Cosine Similarity scoring.
*   **GraphRAG**: A hybrid RAG pipeline that ingests raw documents, extracts triples to populate the `TripleStore`, retrieves semantic text chunks, and traverses multi-hop adjacent paths to construct a grounded, factually anchored context.
*   **Cognitive Memory**: An associative agentic `MemoryGraph` simulating episodic memory networks with temporal decay.
*   **Agent Workflows**: A `TaskExecutionGraph` that takes tool dependencies, topologically schedules tasks, and routes data payloads dynamically.

---

## 🎨 Interactive Glassmorphic Sandbox Dashboard

The repository includes a premium, client-side **Glassmorphic Graph Dashboard Sandbox** to visually interact with the mathematical engines in real-time:
*   **Interactive Force-Directed Physics**: Click to spawn nodes, drag to reposition, and drag-and-connect to create edges. The nodes dynamically organize themselves using Coulomb and Hooke forces.
*   **Real-Time Algorithmic Visualizations**:
    *   *Dijkstra*: Watch the pathfinder expand nodes and highlight the shortest path in an emerald glow.
    *   *PageRank*: Watch energy pulses travel along edges while node radii scale dynamically to represent their relative rank.
    *   *GCN message passing*: Observe feature packets migrate along edges, aggregate at target nodes, and trigger state flashes.
    *   *Spectral Partitioning*: Watch a live power iteration solver compute the Laplacian's second-smallest eigenvector (Fiedler vector), cluster the graph, and draw dashed red cut lines between communities.

To open the dashboard, simply launch **`dashboard/index.html`** in any modern web browser!

---

## 📚 Masterclass PDF Textbook

We compile a publication-quality, 30+ page textbook containing styled mathematical derivations (PageRank convergence, Fiedler Partitioning, GAT attention, TransE margin loss, VGAE ELBO), complexity matrices, and SVG diagrams directly from the source.

*   **PDF Filename**: `Graph_Ecosystem_Masterclass.pdf`
*   **Compiler Script**: `generate_masterclass_pdf.py` (uses headless Google Chrome to print the HTML template).

---

## 🚀 Quick Start & Installation

### 1. Clone & Install Dependencies
Ensure you have Python 3.8+ and PyTorch installed.
```bash
git clone https://github.com/AshishIITD/Graph.git
cd Graph
pip install -r requirements.txt
```

### 2. Run the Automated Verification Suite
Run the master runner to execute unit tests and demonstrations for all 6 phases:
```bash
python3 run_demo.py
```

### 3. Compile the Textbook PDF
Generate the technical master PDF:
```bash
python3 generate_masterclass_pdf.py
```

### 4. Launch the Sandbox UI
Double-click or open `dashboard/index.html` in your browser to play with the interactive sandbox!

---

## 📁 Repository Structure
```text
Graph/
├── dashboard/                 # Glassmorphic Front-end Sandbox
│   ├── index.html             # UI Structure
│   ├── style.css              # Glassmorphic Stylesheet
│   └── app.js                 # Force Physics & Algo Visualizer
├── src/                       # Masterclass Source Package
│   ├── __init__.py            
│   ├── phase1_foundations.py  # Centrality & Spectral Theory
│   ├── phase2_algorithms.py   # Canonical Solvers & Flows
│   ├── phase3_embeddings_pgm.py # Node2Vec & PGMs (HMM, Belief Prop)
│   ├── phase4_knowledge_graphs.py # TripleStore & PyTorch KGE Models
│   ├── phase5_gnns.py         # MessagePassing, GCN, GAT, VGAE
│   └── phase6_graphrag_agents.py # GraphRAG & Agent Task DAGs
├── run_demo.py                # Automated Verification Suite
├── generate_masterclass_pdf.py # PDF Textbook Compiler
├── Graph_Ecosystem_Masterclass.pdf # Finished Textbook
├── requirements.txt           # Python Dependencies
└── .gitignore                 # Version Control Ignore List
```

---

## ⚖️ License
This repository is licensed under the MIT License. Built for pedagogical transparency and production-grade graph engineering reference.
