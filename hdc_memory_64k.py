"""
HDC Memory System - 64k Dimensions with Binary Vectors and Bundle Merging

Upgrades from 10k version:
- 64,000 dimensions for higher precision regime detection
- Strict bipolar vectors (-1, +1) for memory efficiency
- Bundle merging for 1-shot memory updates without retraining
- Hierarchical bundles for extended capacity
- Optimized similarity computation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import json
import os

# Optional: Use sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class Memory:
    """Single memory unit with rich metadata"""
    text: str
    timestamp: float
    memory_type: str  # 'user_input', 'clara_response', 'preference', 'fact', 'episode'
    importance: float = 0.5
    domain: str = "general"
    entities: List[str] = field(default_factory=list)
    turn_id: int = 0
    bundle_id: Optional[int] = None  # Which bundle this memory belongs to


class BinaryHDC64k:
    """
    High-dimensional binary vector operations for 64k dimensions.

    All vectors are strictly bipolar: {-1, +1}
    This enables:
    - Efficient storage (can use int8 or even bit-packed)
    - XOR-based binding (element-wise multiply for bipolar)
    - Majority-vote bundling
    - High noise resistance
    """

    def __init__(self, dim: int = 64000, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def random_hv(self) -> np.ndarray:
        """Generate random bipolar hypervector {-1, +1}"""
        return self.rng.choice([-1, 1], size=self.dim).astype(np.int8)

    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (element-wise XOR for bipolar = multiplication)
        Creates associations: bind(A, B) is dissimilar to both A and B
        """
        return (hv1 * hv2).astype(np.int8)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind (inverse of bind for bipolar vectors)
        unbind(bind(A, B), B) ≈ A
        """
        return self.bind(bound, key)  # XOR is self-inverse

    def bundle(self, hvs: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle multiple hypervectors via weighted majority vote.

        The result is similar to all inputs (superposition).
        Capacity: ~sqrt(dim) vectors before interference (~253 for 64k)
        """
        if not hvs:
            return np.zeros(self.dim, dtype=np.int8)

        if len(hvs) == 1:
            return hvs[0].copy()

        if weights is None:
            weights = [1.0] * len(hvs)

        # Weighted sum
        weighted_sum = np.zeros(self.dim, dtype=np.float32)
        for hv, w in zip(hvs, weights):
            weighted_sum += hv.astype(np.float32) * w

        # Majority vote (sign function)
        result = np.sign(weighted_sum).astype(np.int8)

        # Handle ties randomly
        ties = result == 0
        if np.any(ties):
            result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))

        return result

    def bundle_merge(self, existing: np.ndarray, new_hv: np.ndarray,
                     existing_weight: float = 0.9) -> np.ndarray:
        """
        Merge a new vector into existing bundle (1-shot update).

        This allows adding new memories without rebuilding the entire index.
        The existing_weight controls how much the old bundle dominates.
        """
        new_weight = 1.0 - existing_weight
        return self.bundle([existing, new_hv], [existing_weight, new_weight])

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Cosine similarity for bipolar vectors.
        Equivalent to: (matching_bits - mismatching_bits) / total_bits
        Range: [-1, 1]
        """
        # For bipolar, dot product / dim gives normalized similarity
        return float(np.dot(hv1.astype(np.float32), hv2.astype(np.float32))) / self.dim

    def hamming_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Hamming similarity (fraction of matching elements).
        Range: [0, 1]
        """
        matches = np.sum(hv1 == hv2)
        return float(matches) / self.dim

    def permute(self, hv: np.ndarray, n: int = 1) -> np.ndarray:
        """Cyclic permutation - used for sequence encoding"""
        return np.roll(hv, n)


class HDCMemory64k:
    """
    Hyperdimensional Computing Memory System - 64k Dimensions

    Features:
    - 64,000-dimensional bipolar vectors
    - Neural embedding → HDC projection
    - Bundle merging for 1-shot memory updates
    - Hierarchical bundles for extended capacity
    - Structured role-filler bindings
    - Efficient similarity search

    Memory Capacity:
    - Individual memories: unlimited (stored separately)
    - Per-bundle superposition: ~250 items (sqrt(64000))
    - Hierarchical: unlimited via bundle-of-bundles
    """

    # Constants
    DEFAULT_DIM = 64000
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
    RECALL_THRESHOLD = 0.12  # Lower threshold due to higher dimensionality
    MEMORY_BOOST_THRESHOLD = 0.15
    BUNDLE_CAPACITY = 200  # Conservative estimate for bundle capacity

    def __init__(self,
                 dim: int = DEFAULT_DIM,
                 embedder=None,
                 seed: int = 42,
                 debug: bool = False):
        """
        Initialize HDC Memory with 64k dimensions.

        Args:
            dim: Hypervector dimension (default 64000)
            embedder: Optional sentence transformer (will load if None)
            seed: Random seed for reproducibility
            debug: Enable debug output
        """
        self.dim = dim
        self.debug = debug
        self.seed = seed
        self.current_turn = 0

        # Core HDC operations
        self.hdc = BinaryHDC64k(dim=dim, seed=seed)
        self.rng = np.random.RandomState(seed)

        # Embedding model
        if embedder is not None:
            self.embedder = embedder
            self._external_embedder = True
        elif HAS_SENTENCE_TRANSFORMERS:
            print(f"   Loading embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self._external_embedder = False
        else:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        # Random projection matrix: 384-dim → 64k-dim
        print(f"   Initializing projection matrix ({self.EMBEDDING_DIM} → {dim})...")
        self.projection = self.rng.randn(self.EMBEDDING_DIM, dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

        # Memory stores
        self.memories: List[Tuple[np.ndarray, Memory]] = []

        # Bundle hierarchy for efficient retrieval
        self.memory_bundle = np.zeros(dim, dtype=np.int8)  # Global bundle
        self.domain_bundles: Dict[str, np.ndarray] = {}    # Per-domain bundles
        self.bundle_counts: Dict[str, int] = {"global": 0} # Track items per bundle

        # Symbol library for structured binding
        print(f"   Building symbol library...")
        self.symbols: Dict[str, np.ndarray] = {}
        self._init_symbols()

        # Entity index for fast lookup
        self.entity_index: Dict[str, List[int]] = {}

        # Personality vectors
        print(f"   Encoding personality vectors...")
        self.personality = self._init_personality()

        print(f"   HDC Memory 64k initialized (dim={dim:,})")
        print(f"   Memory per vector: {dim * 1 / 1024:.1f} KB (int8)")
        print(f"   Bundle capacity: ~{self.BUNDLE_CAPACITY} items")

    def _init_symbols(self):
        """Create base symbol vocabulary"""
        base_symbols = [
            # Roles
            "ROLE_USER", "ROLE_CLARA", "ROLE_TOPIC", "ROLE_OUTCOME", "ROLE_ENTITY",
            "ROLE_QUERY", "ROLE_RESPONSE", "ROLE_CONTEXT",
            # Actions
            "ASKED", "ANSWERED", "OFFERED", "RECEIVED", "DISCUSSED", "EXPLAINED",
            # Domains
            "MEDICAL", "CODING", "TEACHING", "QUANTUM", "PERSONALITY", "GENERAL",
            # Conversational entities
            "COFFEE", "TEA", "FOOD", "DRINK", "HELP", "THANKS", "PROJECT", "CODE",
            # Outcomes
            "HELPFUL", "CONFUSED", "SATISFIED", "FRUSTRATED", "SUCCESS", "FAILURE",
            # Time markers
            "RECENT", "TODAY", "THIS_SESSION", "EARLIER", "PREVIOUS"
        ]
        for s in base_symbols:
            self.symbols[s] = self.hdc.random_hv()

    def _init_personality(self) -> Dict[str, np.ndarray]:
        """Encode Clara's personality traits as hypervectors"""
        traits = {
            'warmth': 0.85,
            'curiosity': 0.75,
            'patience': 0.90,
            'encouragement': 0.80,
        }

        personality = {}
        for trait, strength in traits.items():
            base_hv = self.hdc.random_hv()
            # Scale by strength (will be binarized in bundle)
            personality[trait] = base_hv

        # Composite personality vector
        trait_hvs = list(personality.values())
        trait_weights = list(traits.values())
        personality['composite'] = self.hdc.bundle(trait_hvs, trait_weights)

        return personality

    def _text_to_hv(self, text: str) -> np.ndarray:
        """Convert text to binary hypervector via embedding projection"""
        # Get neural embedding
        embedding = self.embedder.encode(text)

        # Project to high dimension
        projected = embedding @ self.projection

        # Binarize to bipolar
        hv = np.sign(projected).astype(np.int8)

        # Handle zeros (rare but possible)
        zeros = hv == 0
        if np.any(zeros):
            hv[zeros] = self.rng.choice([-1, 1], size=np.sum(zeros))

        return hv

    def _get_symbol(self, name: str) -> np.ndarray:
        """Get or create a symbol hypervector"""
        name_upper = name.upper().replace(" ", "_")
        if name_upper not in self.symbols:
            self.symbols[name_upper] = self.hdc.random_hv()
        return self.symbols[name_upper]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text for indexing"""
        import re

        entity_patterns = [
            r'\bcoffee\b', r'\btea\b', r'\bwater\b', r'\bdrink\b',
            r'\bfood\b', r'\blunch\b', r'\bdinner\b', r'\bbreakfast\b',
            r'\bproject\b', r'\bwork\b', r'\bmeeting\b', r'\btask\b',
            r'\bDocker\b', r'\bPython\b', r'\bcode\b', r'\bnotebook\b',
            r'\bhelp\b', r'\bthanks\b', r'\bplease\b',
            r'\berror\b', r'\bbug\b', r'\bfix\b', r'\btest\b',
        ]

        entities = []
        text_lower = text.lower()

        for pattern in entity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities.append(match.group().upper())

        return list(set(entities))

    # === Core Memory Operations ===

    def store(self, text: str,
              memory_type: str = "user_input",
              importance: float = 0.5,
              domain: str = "general",
              increment_turn: bool = False,
              **bindings) -> int:
        """
        Store a memory with entity extraction and structural bindings.

        Uses bundle merging for 1-shot update to global and domain bundles.

        Args:
            text: The memory content
            memory_type: 'user_input', 'clara_response', 'preference', 'fact', 'episode'
            importance: 0.0-1.0 importance score
            domain: Domain category
            increment_turn: Start new conversation turn
            **bindings: Structural role-filler pairs

        Returns:
            Index of stored memory
        """
        if increment_turn:
            self.current_turn += 1

        # Extract entities
        entities = self._extract_entities(text)

        # Create semantic hypervector from text
        text_hv = self._text_to_hv(text)

        # Add domain binding
        domain_hv = self._get_symbol(domain.upper())
        text_hv = self.hdc.bind(text_hv, domain_hv)

        # Add entity bindings
        if entities:
            entity_hvs = []
            role_entity = self._get_symbol("ROLE_ENTITY")
            for entity in entities:
                entity_hv = self._get_symbol(entity)
                entity_hvs.append(self.hdc.bind(role_entity, entity_hv))
            if entity_hvs:
                entity_bundle = self.hdc.bundle(entity_hvs)
                text_hv = self.hdc.bundle([text_hv, entity_bundle])

        # Add structural bindings
        if bindings:
            bound_parts = []
            for role, filler in bindings.items():
                role_hv = self._get_symbol(f"ROLE_{role.upper()}")
                filler_hv = self._get_symbol(str(filler).upper())
                bound_parts.append(self.hdc.bind(role_hv, filler_hv))
            if bound_parts:
                structure_hv = self.hdc.bundle(bound_parts)
                text_hv = self.hdc.bundle([text_hv, structure_hv])

        # Create memory object
        memory = Memory(
            text=text,
            timestamp=time.time(),
            memory_type=memory_type,
            importance=importance,
            domain=domain,
            entities=entities,
            turn_id=self.current_turn
        )

        # Store individual memory
        memory_idx = len(self.memories)
        self.memories.append((text_hv, memory))

        # Update entity index
        for entity in entities:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(memory_idx)

        # Bundle merge into global bundle (1-shot update!)
        self.bundle_counts["global"] = self.bundle_counts.get("global", 0) + 1
        weight = self._get_bundle_weight(self.bundle_counts["global"])
        self.memory_bundle = self.hdc.bundle_merge(self.memory_bundle, text_hv, weight)

        # Bundle merge into domain-specific bundle
        if domain not in self.domain_bundles:
            self.domain_bundles[domain] = text_hv.copy()
            self.bundle_counts[domain] = 1
        else:
            self.bundle_counts[domain] += 1
            weight = self._get_bundle_weight(self.bundle_counts[domain])
            self.domain_bundles[domain] = self.hdc.bundle_merge(
                self.domain_bundles[domain], text_hv, weight
            )

        if self.debug:
            print(f"      [MEM] Stored: '{text[:50]}...'")
            print(f"            Entities: {entities}, Domain: {domain}")
            print(f"            Turn: {self.current_turn}, Idx: {memory_idx}")

        return memory_idx

    def _get_bundle_weight(self, count: int) -> float:
        """
        Calculate bundle weight based on item count.
        Decreases weight for new items as bundle grows to prevent dilution.
        """
        if count <= 1:
            return 0.5
        elif count < 10:
            return 0.8
        elif count < 50:
            return 0.9
        elif count < self.BUNDLE_CAPACITY:
            return 0.95
        else:
            return 0.98  # Very conservative for large bundles

    def recall(self, query: str,
               top_k: int = 5,
               domain_filter: Optional[str] = None,
               include_entities: bool = True,
               threshold: Optional[float] = None) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories similar to query.

        Args:
            query: Search query
            top_k: Number of results
            domain_filter: Optional domain constraint
            include_entities: Boost for shared entities
            threshold: Minimum similarity (default: RECALL_THRESHOLD)

        Returns:
            List of (Memory, score) tuples
        """
        if not self.memories:
            return []

        threshold = threshold or self.RECALL_THRESHOLD
        query_hv = self._text_to_hv(query)
        query_entities = self._extract_entities(query) if include_entities else []

        if self.debug:
            print(f"      [RECALL] Query: '{query[:40]}...'")
            print(f"               Entities: {query_entities}")

        results = []
        for idx, (hv, memory) in enumerate(self.memories):
            # Apply domain filter
            if domain_filter and memory.domain != domain_filter:
                continue

            # Base similarity
            sim = self.hdc.similarity(query_hv, hv)

            # Entity overlap boost
            entity_boost = 0.0
            if query_entities and memory.entities:
                overlap = set(query_entities) & set(memory.entities)
                if overlap:
                    entity_boost = 0.12 * len(overlap)
                    if self.debug:
                        print(f"               Entity match: {overlap} (+{entity_boost:.2f})")

            # Recency boost (turn-based)
            turn_diff = self.current_turn - memory.turn_id
            recency = 1.0 / (1.0 + turn_diff * 0.1)

            # Time-based recency
            age_hours = (time.time() - memory.timestamp) / 3600
            time_recency = 1.0 / (1.0 + age_hours / 24)

            # Combined score
            weighted = (
                sim + entity_boost
            ) * (0.6 + 0.2 * memory.importance) * (0.7 + 0.15 * recency + 0.15 * time_recency)

            if weighted >= threshold or sim >= threshold:
                results.append((memory, weighted))

                if self.debug and sim > 0.08:
                    print(f"               [{idx}] sim={sim:.3f} final={weighted:.3f} '{memory.text[:30]}...'")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recall_by_domain(self, query: str, domain: str, top_k: int = 3) -> List[Tuple[Memory, float]]:
        """
        Fast retrieval using domain-specific bundle.
        First checks bundle similarity, then searches within domain.
        """
        if domain not in self.domain_bundles:
            return []

        query_hv = self._text_to_hv(query)

        # Check domain relevance via bundle
        bundle_sim = self.hdc.similarity(query_hv, self.domain_bundles[domain])

        if bundle_sim < 0.05:  # Query not relevant to domain
            return []

        # Search within domain
        return self.recall(query, top_k=top_k, domain_filter=domain)

    def recall_by_entity(self, entity: str) -> List[Memory]:
        """Fast lookup by entity name"""
        entity_upper = entity.upper()
        if entity_upper not in self.entity_index:
            return []

        indices = self.entity_index[entity_upper]
        return [self.memories[i][1] for i in indices]

    def get_context_for_routing(self, query: str) -> np.ndarray:
        """Get memory-augmented context hypervector for routing"""
        query_hv = self._text_to_hv(query)

        relevant = self.recall(query, top_k=3)
        if relevant:
            memory_hvs = [self._text_to_hv(m.text) for m, _ in relevant]
            memory_context = self.hdc.bundle(memory_hvs)
        else:
            memory_context = np.zeros(self.dim, dtype=np.int8)

        # Combine query + memory context + personality
        routing_hv = self.hdc.bundle(
            [query_hv, memory_context, self.personality['composite']],
            [1.0, 0.3, 0.2]
        )

        return routing_hv

    def get_context_string(self, query: str, max_memories: int = 3) -> str:
        """Get memory context as string for prompt injection"""
        relevant = self.recall(query, top_k=max_memories)
        if not relevant:
            return ""

        good_memories = [(m, s) for m, s in relevant if s > self.RECALL_THRESHOLD]

        if self.debug:
            print(f"      [CONTEXT] {len(relevant)} retrieved, {len(good_memories)} above threshold")

        if not good_memories:
            return ""

        context_parts = ["[Conversation context:"]
        for mem, score in good_memories:
            if mem.memory_type == "user_input":
                context_parts.append(f"- User said: {mem.text[:100]}")
            elif mem.memory_type == "clara_response":
                context_parts.append(f"- Clara said: {mem.text[:100]}")
            else:
                context_parts.append(f"- {mem.text[:100]}")
        context_parts.append("]")

        return "\n".join(context_parts)

    def get_recent_context(self, n_turns: int = 2) -> str:
        """Get context from most recent N conversation turns"""
        if not self.memories:
            return ""

        min_turn = max(0, self.current_turn - n_turns)
        recent = [m for _, m in self.memories if m.turn_id >= min_turn]

        if not recent:
            return ""

        context_parts = ["[Recent conversation:"]
        for mem in recent[-6:]:
            if mem.memory_type == "user_input":
                context_parts.append(f"- User: {mem.text[:80]}")
            elif mem.memory_type == "clara_response":
                context_parts.append(f"- Clara: {mem.text[:80]}")
        context_parts.append("]")

        return "\n".join(context_parts)

    # === Statistics & Persistence ===

    def stats(self) -> Dict:
        """Get memory statistics"""
        domain_counts = {}
        type_counts = {}
        for _, m in self.memories:
            domain_counts[m.domain] = domain_counts.get(m.domain, 0) + 1
            type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1

        return {
            "total_memories": len(self.memories),
            "dimensions": self.dim,
            "by_domain": domain_counts,
            "by_type": type_counts,
            "symbols": len(self.symbols),
            "entities_tracked": len(self.entity_index),
            "current_turn": self.current_turn,
            "bundle_counts": self.bundle_counts.copy(),
            "size_kb": self.size_bytes() / 1024,
            "vector_size_bytes": self.dim,  # int8 = 1 byte
        }

    def size_bytes(self) -> int:
        """Memory footprint (excluding embedder)"""
        # Projection matrix (float32)
        projection_size = self.projection.nbytes

        # Individual memories (int8)
        memories_size = sum(hv.nbytes for hv, _ in self.memories)

        # Bundles (int8)
        bundle_size = self.memory_bundle.nbytes
        domain_bundle_size = sum(b.nbytes for b in self.domain_bundles.values())

        # Symbols (int8)
        symbols_size = sum(hv.nbytes for hv in self.symbols.values())

        # Personality (int8)
        personality_size = sum(hv.nbytes for hv in self.personality.values())

        return projection_size + memories_size + bundle_size + domain_bundle_size + symbols_size + personality_size

    def save(self, path: str):
        """Save memory state to file"""
        data = {
            'version': '3.0-64k',
            'dim': self.dim,
            'seed': self.seed,
            'current_turn': self.current_turn,
            'recall_threshold': self.RECALL_THRESHOLD,
            'memories': [
                {
                    'hv': hv.tolist(),
                    'text': m.text,
                    'timestamp': m.timestamp,
                    'memory_type': m.memory_type,
                    'importance': m.importance,
                    'domain': m.domain,
                    'entities': m.entities,
                    'turn_id': m.turn_id
                }
                for hv, m in self.memories
            ],
            'symbols': {k: v.tolist() for k, v in self.symbols.items()},
            'entity_index': self.entity_index,
            'memory_bundle': self.memory_bundle.tolist(),
            'domain_bundles': {k: v.tolist() for k, v in self.domain_bundles.items()},
            'bundle_counts': self.bundle_counts,
            'projection': self.projection.tolist()
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        size_kb = os.path.getsize(path) / 1024
        print(f"   Saved {len(self.memories)} memories to {path}")
        print(f"   File size: {size_kb:.1f} KB")

    def load(self, path: str) -> bool:
        """Load memory state from file"""
        if not os.path.exists(path):
            print(f"   No memory file found at {path}")
            return False

        with open(path) as f:
            data = json.load(f)

        version = data.get('version', '1.0')
        print(f"   Loading memory file version {version}")

        if data.get('dim') != self.dim:
            print(f"   Dimension mismatch: file has {data.get('dim')}, expected {self.dim}")
            return False

        # Load memories
        self.memories = []
        for m in data['memories']:
            memory = Memory(
                text=m['text'],
                timestamp=m['timestamp'],
                memory_type=m.get('memory_type', 'interaction'),
                importance=m['importance'],
                domain=m.get('domain', 'general'),
                entities=m.get('entities', []),
                turn_id=m.get('turn_id', 0)
            )
            hv = np.array(m['hv'], dtype=np.int8)
            self.memories.append((hv, memory))

        # Load other state
        self.symbols = {k: np.array(v, dtype=np.int8) for k, v in data['symbols'].items()}
        self.entity_index = data.get('entity_index', {})
        self.memory_bundle = np.array(data['memory_bundle'], dtype=np.int8)
        self.current_turn = data.get('current_turn', 0)
        self.bundle_counts = data.get('bundle_counts', {"global": len(self.memories)})

        # Load domain bundles
        if 'domain_bundles' in data:
            self.domain_bundles = {
                k: np.array(v, dtype=np.int8)
                for k, v in data['domain_bundles'].items()
            }

        # Load projection matrix
        if 'projection' in data:
            self.projection = np.array(data['projection'], dtype=np.float32)

        print(f"   Loaded {len(self.memories)} memories from {path}")
        return True


# === Demo / Test ===

def demo_hdc_64k():
    """Demonstrate 64k HDC memory capabilities"""

    print("=" * 70)
    print("HDC MEMORY 64k - DEMONSTRATION")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing 64k HDC Memory...")
    memory = HDCMemory64k(dim=64000, debug=True)

    print(f"\nStats:")
    stats = memory.stats()
    print(f"   Dimensions: {stats['dimensions']:,}")
    print(f"   Vector size: {stats['vector_size_bytes']:,} bytes ({stats['vector_size_bytes']/1024:.1f} KB)")

    # Store memories
    print("\n2. Storing memories with bundle merging...")

    memories_to_store = [
        ("User asked about Python async programming", "coding", "user_input"),
        ("Clara explained how asyncio event loops work", "coding", "clara_response"),
        ("User mentioned they're feeling stressed about work", "personality", "user_input"),
        ("Clara offered supportive advice about work-life balance", "personality", "clara_response"),
        ("User asked about quantum entanglement", "quantum", "user_input"),
        ("Clara explained superposition and measurement", "quantum", "clara_response"),
        ("User offered Clara some coffee", "personality", "user_input"),
        ("Clara gratefully accepted the coffee offer", "personality", "clara_response"),
    ]

    for text, domain, mem_type in memories_to_store:
        memory.store(text, memory_type=mem_type, domain=domain, increment_turn=(mem_type == "user_input"))

    print(f"\nStored {len(memories_to_store)} memories")
    print(f"Bundle counts: {memory.bundle_counts}")

    # Test recall
    print("\n3. Testing memory recall...")

    queries = [
        "How do I use async in Python?",
        "I'm feeling overwhelmed",
        "How's the coffee?",
        "Tell me about quantum physics",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = memory.recall(query, top_k=2)
        for mem, score in results:
            print(f"      [{score:.3f}] ({mem.domain}) {mem.text[:50]}...")

    # Test domain-specific recall
    print("\n4. Domain-specific recall (coding)...")
    results = memory.recall_by_domain("event loops", "coding", top_k=2)
    for mem, score in results:
        print(f"   [{score:.3f}] {mem.text[:60]}...")

    # Stats
    print("\n5. Final statistics:")
    stats = memory.stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Memory size: {stats['size_kb']:.1f} KB")
    print(f"   Symbols: {stats['symbols']}")
    print(f"   Entities: {stats['entities_tracked']}")
    print(f"   By domain: {stats['by_domain']}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_hdc_64k()
