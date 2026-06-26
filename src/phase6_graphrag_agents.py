"""
Graph Ecosystem Masterclass - Phase 6: Enterprise GraphRAG & Agentic Graph Architectures
========================================================================================
This module implements the cutting edge of Graph AI in 2026. It contains:
1. A from-scratch TF-IDF Vector Store for semantic document retrieval.
2. A complete hybrid GraphRAG Pipeline fusing vector search with multi-hop KG traversal.
3. An associative Agentic Memory Graph simulating episodic cognitive networks.
4. A Task Dependency Execution Graph showing how agents plan and execute tool workflows.
"""

import re
from collections import deque
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Callable
from src.phase1_foundations import Graph
from src.phase2_algorithms import topological_sort
from src.phase4_knowledge_graphs import TripleStore

# ==========================================
# 1. From-Scratch TF-IDF Vector Store
# ==========================================

class SimpleVectorStore:
    """
    A lightweight, from-scratch Vector Database utilizing TF-IDF term-frequency 
    vectorization and Cosine Similarity to perform semantic text retrieval.
    Guarantees local execution without requiring external API keys.
    """
    def __init__(self):
        self.documents: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.tfidf_matrix: np.ndarray = np.array([])
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
        
    def fit_and_index(self, documents: List[str]) -> None:
        """Fits the TF-IDF vectorizer and builds the document index matrix."""
        self.documents = documents
        if not documents:
            return
            
        # 1. Build Vocabulary
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        word_idx = 0
        for doc in tokenized_docs:
            for token in doc:
                if token not in self.vocab:
                    self.vocab[token] = word_idx
                    word_idx += 1
                    
        num_docs = len(documents)
        num_words = len(self.vocab)
        
        # 2. Compute Term Frequencies (TF) and Document Frequencies (DF)
        tf = np.zeros((num_docs, num_words))
        df = np.zeros(num_words)
        
        for d_idx, doc in enumerate(tokenized_docs):
            unique_words = set(doc)
            for token in doc:
                w_idx = self.vocab[token]
                tf[d_idx, w_idx] += 1
            for token in unique_words:
                w_idx = self.vocab[token]
                df[w_idx] += 1
                
        # Normalise term frequencies by document length
        for d_idx in range(num_docs):
            length = len(tokenized_docs[d_idx])
            if length > 0:
                tf[d_idx, :] /= length
                
        # 3. Compute Inverse Document Frequencies (IDF)
        # Smooth IDF: idf = log(1 + N / (1 + df))
        self.idf = np.log(1 + num_docs / (1 + df))
        
        # 4. Build TF-IDF matrix
        self.tfidf_matrix = tf * self.idf
        
    def query(self, query_text: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Retrieves the top_k most semantically similar documents to the query_text.
        Uses Cosine Similarity: Sim(q, d) = (q . d) / (||q|| * ||d||)
        """
        if not self.documents:
            return []
            
        # Vectorize query
        tokens = self._tokenize(query_text)
        q_tf = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                q_tf[self.vocab[token]] += 1
                
        if len(tokens) > 0:
            q_tf /= len(tokens)
            
        q_tfidf = q_tf * self.idf
        
        # Compute Cosine Similarity against all documents
        q_norm = np.linalg.norm(q_tfidf)
        if q_norm == 0:
            # Query contains no words in our vocabulary, return first top_k docs
            return [(self.documents[i], 0.0) for i in range(min(top_k, len(self.documents)))]
            
        scores = []
        for d_idx in range(len(self.documents)):
            doc_tfidf = self.tfidf_matrix[d_idx]
            d_norm = np.linalg.norm(doc_tfidf)
            if d_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(q_tfidf, doc_tfidf) / (q_norm * d_norm)
            scores.append((self.documents[d_idx], float(similarity)))
            
        # Sort by similarity score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==========================================
# 2. Complete GraphRAG Pipeline
# ==========================================

class GraphRAGPipeline:
    """
    Enterprise-grade GraphRAG Pipeline.
    Integrates unstructured documents, extracts entities and relations,
    indexes them into a hybrid Vector + Graph Store, and performs multi-hop path retrieval.
    """
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.triple_store = TripleStore()
        # Mapping: Entity -> Document indices where it was mentioned
        self.entity_sources: Dict[str, Set[int]] = {}
        # Mapping: Document index -> Chunk text
        self.chunks: List[str] = []
        
    def _extract_entities_and_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        A lightweight rule-based semantic parser that extracts subject-predicate-object
        triples from raw text. Looks for common patterns and capitalised entity names.
        """
        triples = []
        # Pattern 1: Capitalized Name (Entity) -> action/verb -> Capitalized Name (Entity)
        # e.g., "Alice works_at Google" or "Google located_in USA"
        pattern = r'\b([A-Z][a-zA-Z]+)\s+([a-z_]+)\s+([A-Z][a-zA-Z\s0-9]+?)(?=\.|\,|and|\s+is\b|\s+was\b|$)'
        matches = re.findall(pattern, text)
        for sub, pred, obj in matches:
            triples.append((sub.strip(), pred.strip(), obj.strip()))
            
        # Pattern 2: Copula relationships (X is a Y)
        copulas = re.findall(r'\b([A-Z][a-zA-Z]+)\s+is\s+(?:a|an)?\s*([a-zA-Z\s]+?)(?=\.|\,|$)', text)
        for entity, relation_type in copulas:
            # Clean relation type
            rel = relation_type.strip().replace(" ", "_")
            triples.append((entity.strip(), "is_a", rel))
            
        return triples

    def ingest_document(self, doc_text: str) -> None:
        """
        Processes document: chunks it, extracts knowledge triples,
        and populates vector store and triple store indexes.
        """
        # Chunking: split by paragraphs/sentences
        paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
        start_idx = len(self.chunks)
        self.chunks.extend(paragraphs)
        
        # Extract and index triples from each chunk
        for i, chunk in enumerate(paragraphs):
            chunk_global_idx = start_idx + i
            triples = self._extract_entities_and_relations(chunk)
            
            for sub, pred, obj in triples:
                self.triple_store.add_triple(sub, pred, obj)
                
                # Update entity sources for grounding lookup
                if sub not in self.entity_sources: self.entity_sources[sub] = set()
                if obj not in self.entity_sources: self.entity_sources[obj] = set()
                
                self.entity_sources[sub].add(chunk_global_idx)
                self.entity_sources[obj].add(chunk_global_idx)
                
        # Update/fit the vector store on all compiled chunks
        self.vector_store.fit_and_index(self.chunks)

    def retrieve(self, query: str, top_k_vector: int = 2, max_hops: int = 2) -> Dict[str, any]:
        """
        Performs Hybrid Graph-Augmented Retrieval.
        1. Queries Vector Store for top_k semantic chunks.
        2. Identifies key entities mentioned in those chunks and the query.
        3. Traverses the Triple Store up to max_hops from these seed entities to retrieve relevant graph paths.
        4. Fuses chunks and graph paths into a structured context.
        """
        # 1. Vector Search
        vector_results = self.vector_store.query(query, top_k=top_k_vector)
        retrieved_chunks = [doc for doc, score in vector_results]
        
        # 2. Identify Seed Entities (words present in both query/chunks and the KG)
        words_in_play = set(re.findall(r'\b\w+\b', query.lower() + " " + " ".join(retrieved_chunks)))
        seed_entities = []
        for ent in self.entity_sources:
            if ent.lower() in words_in_play:
                seed_entities.append(ent)
                
        # 3. Multi-Hop Graph Traversal
        retrieved_triples: Set[Tuple[str, str, str]] = set()
        visited_entities = set(seed_entities)
        queue = deque([(ent, 0) for ent in seed_entities])
        
        while queue:
            curr_ent, hop = queue.popleft()
            if hop >= max_hops:
                continue
                
            # Find all triples connected to this entity
            # Subject matching: (curr_ent, ?p, ?o)
            subj_triples = self.triple_store.query_single_pattern(curr_ent, "?p", "?o")
            for s, p, o in subj_triples:
                retrieved_triples.add((s, p, o))
                if o not in visited_entities:
                    visited_entities.add(o)
                    queue.append((o, hop + 1))
                    
            # Object matching: (?s, ?p, curr_ent)
            obj_triples = self.triple_store.query_single_pattern("?s", "?p", curr_ent)
            for s, p, o in obj_triples:
                retrieved_triples.add((s, p, o))
                if s not in visited_entities:
                    visited_entities.add(s)
                    queue.append((s, hop + 1))
                    
        # 4. Format Results
        return {
            "query": query,
            "vector_chunks": retrieved_chunks,
            "graph_triples": list(retrieved_triples),
            "seeds": seed_entities
        }

    def generate_grounded_answer(self, query: str) -> str:
        """
        Fuses retrieved vector chunks and graph paths, generating a mock grounded response
        to showcase the RAG output. In production, this context is passed to an LLM.
        """
        data = self.retrieve(query)
        
        # Construct grounded context block
        context_chunks = "\n".join([f"- {chunk}" for chunk in data["vector_chunks"]])
        context_graph = "\n".join([f"- ({s}) --[{p}]--> ({o})" for s, p, o in data["graph_triples"]])
        
        # Simulated LLM response utilizing the context
        prompt = f"""
======= GROUNDED CONTEXT =======
[Unstructured Chunks]:
{context_chunks}

[Structured Knowledge Graph Paths]:
{context_graph}
================================

[User Query]: {query}

[GraphRAG Grounded Response]:
"""
        # Simple rule-based mock response generation to prove the retrieval loop
        response = ""
        triples = data["graph_triples"]
        if query.lower().find("where") != -1 or query.lower().find("located") != -1:
            # Look for location triples
            loc_triples = [t for t in triples if t[1] == "located_in" or t[1] == "works_at"]
            if loc_triples:
                s, p, o = loc_triples[0]
                response = f"Based on the Knowledge Graph paths, we know that {s} is associated with {o} via a '{p}' relationship. "
            else:
                response = "I searched the vector chunks and graph relations, but could not find a specific location. "
        else:
            response = "Using hybrid retrieval, I synthesized information from the vector text and graph relationships. "
            
        if data["vector_chunks"]:
            response += f"According to the retrieved documents: '{data['vector_chunks'][0][:80]}...' "
            
        return prompt.strip() + "\n" + response.strip()


# ==========================================
# 3. Agentic Memory Graph
# ==========================================

class MemoryGraph:
    """
    An associative cognitive network representing an Agent's episodic memory.
    Nodes represent events, tasks, or concepts, and edges represent strength of association.
    Edge strengths decay over time unless reinforced.
    """
    def __init__(self):
        self.graph = Graph(directed=False)
        
    def add_episodic_memory(self, event_id: str, description: str, associations: List[str]) -> None:
        """Adds an event node and connects it to associated concepts with strength 1.0."""
        self.graph.add_node(event_id, type="event", description=description)
        for concept in associations:
            self.graph.add_node(concept, type="concept")
            # Create/reinforce link
            if concept in self.graph.adj[event_id]:
                curr_weight = self.graph.adj[event_id][concept]["weight"]
                new_weight = min(1.0, curr_weight + 0.2) # Reinforce association
            else:
                new_weight = 0.5
            self.graph.add_edge(event_id, concept, weight=new_weight)
            
    def decay_memories(self, decay_factor: float = 0.9) -> None:
        """Simulates forgetting: decays all association weights in the memory graph."""
        edges = self.graph.to_edge_list()
        for u, v, weight in edges:
            new_weight = weight * decay_factor
            if new_weight < 0.1:
                # Remove weak connections entirely
                self.graph.remove_edge(u, v)
            else:
                self.graph.add_edge(u, v, weight=new_weight)

    def retrieve_associated_memories(self, concept: str) -> List[Tuple[str, float]]:
        """Retrieves memories associated with a concept, sorted by association strength."""
        if concept not in self.graph.nodes:
            return []
        
        associations = []
        for neighbor, edge_data in self.graph.adj[concept].items():
            if self.graph.nodes[neighbor]["type"] == "event":
                associations.append((neighbor, edge_data["weight"]))
                
        associations.sort(key=lambda x: x[1], reverse=True)
        return associations


# ==========================================
# 4. Agent Planning & Execution Graph
# ==========================================

class TaskExecutionGraph:
    """
    A Task Dependency Execution Graph (DAG) for Agent planning.
    Nodes represent executable tools (Python functions) and edges represent data dependencies.
    Executes tasks in correct topological order, routing outputs from parents to children.
    """
    def __init__(self):
        self.dag = Graph(directed=True)
        # Mapping: task_name -> executable function
        self.registry: Dict[str, Callable] = {}
        # Stores outputs of executed tasks
        self.results: Dict[str, any] = {}
        
    def register_tool(self, name: str, func: callable) -> None:
        self.registry[name] = func
        self.dag.add_node(name)
        
    def add_dependency(self, parent_task: str, child_task: str) -> None:
        """Declares that child_task requires the output of parent_task."""
        self.dag.add_edge(parent_task, child_task)
        
    def execute_plan(self, initial_inputs: Dict[str, any]) -> Dict[str, any]:
        """
        Executes the plan:
        1. Topologically sorts the tasks to resolve dependencies.
        2. Iterates and executes each task.
        3. Passes the results of parent tasks as arguments to child tasks.
        """
        # Resolve order
        execution_order = topological_sort(self.dag)
        self.results = initial_inputs.copy()
        
        for task in execution_order:
            if task in self.registry:
                # Fetch parent outputs (inputs for this task)
                parents = list(self.dag.rev_adj[task].keys())
                
                # Extract argument values from results dictionary
                task_args = {}
                for p in parents:
                    task_args[f"input_from_{p}"] = self.results[p]
                    
                # If it's a root task, it might use initial inputs directly
                if not parents and task in initial_inputs:
                    task_args["initial_input"] = initial_inputs[task]
                    
                # Execute tool
                print(f"[Agent Executor] Executing tool: {task} with args: {list(task_args.keys())}")
                result = self.registry[task](**task_args)
                self.results[task] = result
                
        return self.results
