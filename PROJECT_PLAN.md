# Clara v2 - Embodied AI Project Plan

## Overview

Clara is an embodied AI system with:
- **Dual-brain architecture**: Knowledge (Phi-3) + Personality (Mistral+LoRA)
- **Hyperdimensional Computing (HDC) memory**: 64k-dim bipolar vectors
- **Intelligent routing**: Nemotron-Orchestrator-8B
- **Multi-tier memory stack**: Session cache → HDC → Graph/Vector DB

---

## Memory Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLARA MEMORY STACK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L1: SESSION CACHE (FalkorDB/Redis)     ⚡ <1ms                 │
│  ├─ Current conversation turns                                  │
│  ├─ Active entities & slots                                     │
│  ├─ Hot routing decisions                                       │
│  └─ Working memory window                                       │
│                                                                 │
│  L2: HDC MEMORY (64k bipolar)           🧠 ~5ms                 │
│  ├─ Semantic similarity search                                  │
│  ├─ Associative binding (⊗)  ←── QC bridge                     │
│  ├─ Episode bundles                                             │
│  └─ Domain-specific bundles                                     │
│                                                                 │
│  L3: GRAPH + VECTOR (FalkorDB)          📚 ~20ms                │
│  ├─ Entity relationships (graph)                                │
│  ├─ Long-term episodic memory (vectors)                         │
│  ├─ Concept ontology                                            │
│  └─ Cross-session knowledge                                     │
│                                                                 │
│  L4: STRUCTURED STORE (SQLite)          🗄️ ~50ms               │
│  ├─ User preferences                                            │
│  ├─ Configuration                                               │
│  └─ Audit logs                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
┌─────────────┐
│ L1: Session │ ← Check active entities, recent turns
└─────────────┘
    │
    ▼
┌─────────────┐
│ L2: HDC 64k │ ← Semantic recall via binding/similarity
└─────────────┘
    │ (cache miss or need relationships?)
    ▼
┌─────────────┐
│ L3: FalkorDB│ ← Graph traversal + vector search
└─────────────┘
    │
    ▼
  Router → Brain → Response
    │
    ▼
  Store: L1 (sync), L2 (sync), L3 (async)
```

---

## HDC → Quantum Computing Bridge

### Why HDC as a QC Bridge?

Hyperdimensional Computing provides a **classical approximation** of quantum-like operations that can later migrate to actual quantum hardware. The mathematical structures align:

### Operation Mapping

| HDC Operation | Mathematical Form | Quantum Analog | Description |
|---------------|-------------------|----------------|-------------|
| **Bind (⊗)** | `A ⊗ B = A * B` (element-wise) | Entanglement / Tensor product | Creates composite representation dissimilar to both inputs |
| **Bundle (+)** | `sign(Σ Aᵢ)` | Superposition | Multiple states coexist in single vector |
| **Permute (ρ)** | `roll(A, n)` | Phase rotation | Encodes position/sequence information |
| **Similarity** | `(A · B) / D` | Measurement | "Collapses" to nearest stored pattern |
| **Unbind** | `A ⊗ B ⊗ B = A` | Disentanglement | Recovers original from bound pair |

### Key Parallels

#### 1. Superposition via Bundling
```
Classical:  |state⟩ = α|0⟩ + β|1⟩

HDC:        bundle = sign(w₁·hv₁ + w₂·hv₂ + ... + wₙ·hvₙ)
            - Each hvᵢ is quasi-orthogonal (random in high-D)
            - Weighted sum preserves similarity to all components
            - "Measurement" = find most similar stored pattern
```

#### 2. Entanglement via Binding
```
Quantum:    |ψ⟩ = |A⟩ ⊗ |B⟩  (tensor product, entangled state)

HDC:        bound = bind(A, B) = A * B  (element-wise)
            - Result is dissimilar to both A and B
            - But: unbind(bound, B) ≈ A (recovers original)
            - Creates "associated" representation
```

#### 3. Interference via Similarity
```
Quantum:    Probability amplitudes interfere constructively/destructively

HDC:        Similar patterns reinforce in bundles
            Dissimilar patterns cancel out (noise)
            High-D ensures random vectors are ~orthogonal
```

### The 64k Dimension Choice

```
D = 64,000 dimensions provides:

1. Orthogonality:  E[sim(random₁, random₂)] ≈ 0
                   Var[sim] ≈ 1/D = 1/64000 ≈ 0.0000156

2. Noise resistance: Error tolerance scales with √D
                     √64000 ≈ 253 (vs √10000 ≈ 100)

3. Bundle capacity:  ~√D items before interference
                     ~253 items per bundle

4. Quantum-ready:    Maps to 64k qubit register (future)
```

### Quantum-Inspired Algorithms (Future)

#### Grover-like Search
```python
# Classical HDC approximation of Grover's algorithm
def quantum_inspired_search(query_hv, memory_bundle, iterations=3):
    """
    Amplitude amplification via iterative refinement
    """
    current = query_hv.copy()

    for _ in range(iterations):
        # "Oracle" - identify matching components
        similarities = [hdc.similarity(current, mem) for mem in memories]

        # "Diffusion" - amplify high-similarity, suppress low
        weights = softmax(similarities * temperature)
        current = hdc.bundle(memories, weights)

    return current  # Amplified toward best match
```

#### Quantum Annealing for Optimization
```python
# Energy-based memory consolidation (sleep/dreaming)
def consolidate_memories(memories, temperature_schedule):
    """
    Simulated annealing over HDC space
    Similar memories cluster, redundant ones merge
    """
    for T in temperature_schedule:  # Cool down
        for i, mem in enumerate(memories):
            # Find neighbors
            neighbors = find_similar(mem, threshold=T)

            # Probabilistic merge (Boltzmann)
            if random() < exp(-energy_diff / T):
                memories[i] = bundle(neighbors)

    return deduplicate(memories)
```

### Migration Path to Quantum Hardware

```
Phase 1 (Current):  Classical HDC on CPU/GPU
                    64k float32/int8 vectors

Phase 2 (Near):     Quantum-inspired on classical
                    Tensor network approximations
                    GPU-accelerated similarity search

Phase 3 (Future):   Hybrid classical-quantum
                    Quantum similarity search (Grover)
                    Quantum bundling (superposition)
                    Classical binding (still efficient)

Phase 4 (Far):      Full quantum HDC
                    64k qubit register
                    Native superposition/entanglement
                    Exponential speedup for search
```

---

## Component Status

### Implemented ✅

| Component | File | Status |
|-----------|------|--------|
| HDC Memory 64k | `hdc_memory_64k.py` | ✅ Complete |
| Nemotron Router | `nemotron_router.py` | ✅ Complete |
| Clara v2 Integration | `clara_v2.py` | ✅ Complete |
| Embedding Router | `nemotron_router.py` | ✅ Complete |
| Hybrid Router | `nemotron_router.py` | ✅ Complete |

### In Progress 🔄

| Component | File | Status |
|-----------|------|--------|
| Session Memory (FalkorDB) | `session_memory.py` | 🔄 Next |
| Graph Memory Layer | TBD | 🔄 Planned |

### Planned 📋

| Component | Description | Priority |
|-----------|-------------|----------|
| Voice LoRA Adapter | Fine-tune personality from chat history | High |
| Tool Execution | Actual tool calling via Orchestrator | Medium |
| Sleep/Consolidation | Memory consolidation during idle | Medium |
| Quantum-Inspired Search | Grover-like amplitude amplification | Low |

---

## FalkorDB Integration

### Why FalkorDB?

FalkorDB provides:
1. **Redis-compatible** - Fast key-value for session cache
2. **Graph database** - Cypher queries for relationships
3. **Vector search** - Similarity search for embeddings
4. **Single container** - Simplifies deployment

### Graph Schema (Clara)

```cypher
// Nodes
(:User {id, name, preferences})
(:Entity {name, type, first_seen, last_seen})
(:Memory {id, text, timestamp, domain, importance})
(:Concept {name, domain})
(:Session {id, started, ended})

// Relationships
(:User)-[:HAS_SESSION]->(:Session)
(:Session)-[:CONTAINS]->(:Memory)
(:Memory)-[:MENTIONS]->(:Entity)
(:Memory)-[:RELATES_TO]->(:Concept)
(:Entity)-[:CONNECTED_TO]->(:Entity)
(:Concept)-[:SUBCONCEPT_OF]->(:Concept)
```

### Usage Pattern

```python
# Session cache (Redis protocol)
await falkor.set(f"session:{sid}:turns", json.dumps(turns))
await falkor.get(f"session:{sid}:entities")

# Graph queries (Cypher)
result = await falkor.graph.query("""
    MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $entity})
    WHERE m.timestamp > $since
    RETURN m.text, m.domain
    ORDER BY m.timestamp DESC
    LIMIT 5
""", {"entity": "coffee", "since": yesterday})

# Vector search
similar = await falkor.graph.query("""
    CALL db.idx.vector.queryNodes('Memory', 'embedding', 5, $query_vec)
    YIELD node, score
    RETURN node.text, score
""", {"query_vec": query_embedding})
```

---

## Voice Adapter (Planned)

### Data Requirements
- ~100k tokens of conversational data
- Consistent persona/style throughout
- Mix of topics and emotional tones

### Training Config
```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
}

training = {
    "base_model": "mistralai/Mistral-Nemo-Base-2407",  # or similar
    "epochs": 3-5,
    "lr": 1e-4,
    "batch_size": 4,
}
```

---

## References

- [HDC Tutorial](https://www.hd-computing.com/)
- [Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B)
- [FalkorDB Docs](https://docs.falkordb.com/)
- [ToolOrchestra Paper](https://arxiv.org/abs/2511.21689)
