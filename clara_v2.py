"""
Clara v2 - Integrated Embodied AI with 64k HDC Memory and Nemotron Router

This module combines:
- 64k-dimensional binary HDC memory with bundle merging
- Nemotron-based routing (LLM + embedding hybrid)
- Dual-brain architecture (Knowledge + Personality)
- Future tool-calling capabilities

Architecture:
```
User Query
    │
    ├─────────────────────────────────────────────┐
    │  HDC Memory 64k                             │
    │  - 64k-dim bipolar vectors                  │
    │  - Bundle merging for 1-shot updates        │
    │  - Domain-specific bundles                  │
    │  - Entity tracking                          │
    └─────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────┐
    │  Nemotron Hybrid Router                     │
    │  - Fast embedding pre-filter                │
    │  - LLM fallback for ambiguous cases         │
    │  - Structured JSON routing decisions        │
    │  - Tool-calling scaffold                    │
    └─────────────────────────────────────────────┘
    │
    ├────────────────┬────────────────────────────┐
    │                │                            │
    KNOWLEDGE BRAIN  PERSONALITY BRAIN     TOOL EXECUTOR
    (Phi-3 merged)   (Mistral + LoRA)      (Future)
    ✓ Medical        ✓ Warmth
    ✓ Coding         ✓ Playful
    ✓ Teaching       ✓ Encouragement
    ✓ Quantum
    │                │                            │
    └────────────────┴────────────────────────────┘
    │
    Response + Memory Store
```
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch

# Local imports
from hdc_memory_64k import HDCMemory64k, Memory
from nemotron_router import (
    NemotronRouter, EmbeddingRouter, HybridRouter,
    RoutingDecision, BrainType, RoutingDomain, ToolCall
)


@dataclass
class ClaraConfig:
    """Configuration for Clara v2"""
    # HDC Memory
    hdc_dim: int = 64000
    hdc_seed: int = 42

    # Router
    router_mode: str = "hybrid"  # "llm", "embedding", "hybrid"
    router_model: str = "nvidia/Nemotron-Orchestrator-8B"  # Purpose-built orchestrator
    router_llm_threshold: float = 0.6  # For hybrid mode

    # Models (paths or HF identifiers)
    knowledge_model: str = "clara-knowledge"  # Path to merged Phi-3
    personality_base: str = "mistralai/Mistral-7B-Instruct-v0.3"
    personality_adapters: Dict[str, str] = None  # adapter_name: path

    # Generation
    max_new_tokens_knowledge: int = 300
    max_new_tokens_personality: int = 200
    temperature: float = 0.7

    # Memory
    memory_file: str = "clara_memory_v2.json"
    auto_save_interval: int = 10  # Save every N interactions

    # Debug
    debug: bool = False

    def __post_init__(self):
        if self.personality_adapters is None:
            self.personality_adapters = {
                "warmth": "mistral_warmth",
                "playful": "mistral_playful",
                "encouragement": "mistral_encouragement",
            }


class ClaraV2:
    """
    Clara v2 - Embodied AI with Advanced Memory and Routing

    Features:
    - 64k-dimensional HDC memory with bundle merging
    - Nemotron-based intelligent routing
    - Dual-brain architecture (Knowledge + Personality)
    - Memory-aware response generation
    - Tool-calling scaffold for future expansion
    """

    def __init__(self, config: Optional[ClaraConfig] = None, models_dir: str = None):
        """
        Initialize Clara v2.

        Args:
            config: ClaraConfig instance (uses defaults if None)
            models_dir: Base directory for model files
        """
        self.config = config or ClaraConfig()
        self.models_dir = models_dir or "./models"

        # State
        self.interaction_count = 0
        self._loaded = False

        # Components (lazy loaded)
        self._memory: Optional[HDCMemory64k] = None
        self._router = None
        self._embedder = None
        self._knowledge_model = None
        self._knowledge_tokenizer = None
        self._personality_model = None
        self._personality_tokenizer = None

        print("=" * 60)
        print("CLARA v2 - Embodied AI")
        print("=" * 60)
        print(f"   HDC Dimensions: {self.config.hdc_dim:,}")
        print(f"   Router Mode: {self.config.router_mode}")
        print(f"   Debug: {self.config.debug}")
        print("=" * 60)

    # === Lazy Loading ===

    @property
    def memory(self) -> HDCMemory64k:
        """Lazy load HDC memory"""
        if self._memory is None:
            print("\n[Clara] Initializing HDC Memory 64k...")

            # Load embedder first (shared)
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

            self._memory = HDCMemory64k(
                dim=self.config.hdc_dim,
                embedder=self._embedder,
                seed=self.config.hdc_seed,
                debug=self.config.debug
            )

            # Try to load existing memory
            memory_path = os.path.join(self.models_dir, "..", self.config.memory_file)
            if os.path.exists(memory_path):
                print(f"[Clara] Loading existing memory from {memory_path}")
                self._memory.load(memory_path)

        return self._memory

    @property
    def router(self):
        """Lazy load router"""
        if self._router is None:
            print(f"\n[Clara] Initializing {self.config.router_mode} router...")

            # Load embedder first (shared)
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

            if self.config.router_mode == "embedding":
                self._router = EmbeddingRouter(
                    embedder=self._embedder,
                    debug=self.config.debug
                )
            elif self.config.router_mode == "llm":
                self._router = NemotronRouter(
                    model_name=self.config.router_model,
                    debug=self.config.debug
                )
            else:  # hybrid
                emb_router = EmbeddingRouter(
                    embedder=self._embedder,
                    debug=self.config.debug
                )
                self._router = HybridRouter(
                    embedding_router=emb_router,
                    llm_threshold=self.config.router_llm_threshold,
                    debug=self.config.debug
                )
                # Set LLM model name for lazy loading
                self._router._llm_model_name = self.config.router_model

        return self._router

    def load_brains(self, knowledge: bool = True, personality: bool = True):
        """
        Load the brain models.

        Args:
            knowledge: Load knowledge brain (Phi-3)
            personality: Load personality brain (Mistral + adapters)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if knowledge and self._knowledge_model is None:
            print("\n[Clara] Loading Knowledge Brain...")
            knowledge_path = os.path.join(self.models_dir, self.config.knowledge_model)

            self._knowledge_tokenizer = AutoTokenizer.from_pretrained(
                knowledge_path,
                trust_remote_code=True
            )

            self._knowledge_model = AutoModelForCausalLM.from_pretrained(
                knowledge_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self._knowledge_model.eval()
            print("   Knowledge Brain loaded")

        if personality and self._personality_model is None:
            print("\n[Clara] Loading Personality Brain...")
            from peft import PeftModel

            self._personality_tokenizer = AutoTokenizer.from_pretrained(
                self.config.personality_base
            )

            personality_base = AutoModelForCausalLM.from_pretrained(
                self.config.personality_base,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # Load first adapter
            adapters = list(self.config.personality_adapters.items())
            first_name, first_path = adapters[0]
            adapter_full_path = os.path.join(self.models_dir, first_path)

            self._personality_model = PeftModel.from_pretrained(
                personality_base,
                adapter_full_path,
                adapter_name=first_name
            )

            # Load remaining adapters
            for adapter_name, adapter_path in adapters[1:]:
                full_path = os.path.join(self.models_dir, adapter_path)
                if os.path.exists(full_path):
                    self._personality_model.load_adapter(full_path, adapter_name=adapter_name)
                    print(f"   Loaded adapter: {adapter_name}")

            self._personality_model.set_adapter("warmth")  # Default
            self._personality_model.eval()
            print("   Personality Brain loaded")

        self._loaded = True

    # === Main Interface ===

    def __call__(self, query: str, **kwargs) -> str:
        """Main interface - same as chat()"""
        return self.chat(query, **kwargs)

    def chat(self,
             query: str,
             use_memory: bool = True,
             store_interaction: bool = True,
             verbose: bool = None) -> str:
        """
        Main chat interface.

        Args:
            query: User's message
            use_memory: Use memory context for routing and generation
            store_interaction: Store this interaction in memory
            verbose: Override config.debug for this call

        Returns:
            Clara's response
        """
        verbose = verbose if verbose is not None else self.config.debug

        # 1. Get memory context
        memory_context = ""
        if use_memory:
            memory_context = self.memory.get_context_string(query)
            if not memory_context and self.memory.current_turn > 0:
                memory_context = self.memory.get_recent_context(n_turns=2)

            if verbose and memory_context:
                print(f"[Clara] Memory context: {len(memory_context)} chars")

        # 2. Route the query
        decision = self.router.route(query, memory_context)

        if verbose:
            print(f"[Clara] Routing: {decision.brain.value}/{decision.domain.value} "
                  f"(conf: {decision.confidence:.2f})")

        # 3. Generate response
        if decision.brain == BrainType.KNOWLEDGE:
            response = self._generate_knowledge(query, decision.domain, memory_context)
        elif decision.brain == BrainType.TOOL:
            response = self._handle_tool_call(query, decision)
        else:
            response = self._generate_personality(query, decision.domain, memory_context)

        # 4. Store interaction in memory
        if store_interaction:
            effective_domain = decision.domain.value

            # Store user query
            self.memory.store(
                text=query,
                memory_type="user_input",
                importance=0.5 + (decision.confidence * 0.3),
                domain=effective_domain,
                increment_turn=True
            )

            # Store Clara's response (truncated)
            self.memory.store(
                text=response[:200],
                memory_type="clara_response",
                importance=0.3,
                domain=effective_domain
            )

            self.interaction_count += 1

            # Auto-save
            if self.interaction_count % self.config.auto_save_interval == 0:
                self.save_memory()

        return response

    def _generate_knowledge(self,
                           query: str,
                           domain: RoutingDomain,
                           memory_context: str) -> str:
        """Generate response using knowledge brain"""
        if self._knowledge_model is None:
            return f"[Knowledge brain not loaded. Domain: {domain.value}]"

        # Build prompt
        if memory_context:
            prompt = f"### Instruction:\n{memory_context}\n\nUser question: {query}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{query}\n\n### Response:\n"

        inputs = self._knowledge_tokenizer(prompt, return_tensors="pt").to(self._knowledge_model.device)

        with torch.no_grad():
            outputs = self._knowledge_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens_knowledge,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self._knowledge_tokenizer.eos_token_id,
                use_cache=False
            )

        response = self._knowledge_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return self._clean_response(response)

    def _generate_personality(self,
                             query: str,
                             domain: RoutingDomain,
                             memory_context: str) -> str:
        """Generate response using personality brain"""
        if self._personality_model is None:
            return f"[Personality brain not loaded. Domain: {domain.value}]"

        # Select adapter based on domain/mood (default to warmth)
        adapter_name = "warmth"
        if domain == RoutingDomain.PERSONALITY:
            # Could add mood detection here
            adapter_name = "warmth"

        self._personality_model.set_adapter(adapter_name)

        # Build prompt
        if memory_context:
            prompt = f"### Instruction:\n{memory_context}\n\nUser message: {query}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{query}\n\n### Response:\n"

        inputs = self._personality_tokenizer(prompt, return_tensors="pt").to(self._personality_model.device)

        with torch.no_grad():
            outputs = self._personality_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens_personality,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self._personality_tokenizer.eos_token_id
            )

        response = self._personality_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return self._clean_response(response)

    def _handle_tool_call(self, query: str, decision: RoutingDecision) -> str:
        """Handle tool call routing (future capability)"""
        # Check if router supports tool calling
        if isinstance(self.router, (NemotronRouter, HybridRouter)):
            if hasattr(self.router, 'llm_router') and self.router.llm_router:
                tool_call = self.router.llm_router.check_tool_call(query)
                if tool_call:
                    return f"[Tool call detected: {tool_call.tool_name}({tool_call.arguments})]"

        # Fallback to personality
        return self._generate_personality(query, RoutingDomain.PERSONALITY, "")

    def _clean_response(self, response: str) -> str:
        """Clean up response artifacts"""
        stop_markers = [
            "### Instruction:", "Instruction:", "\n\n\n",
            "User:", "[Conversation context:", "###"
        ]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        return response.strip()

    # === Memory Operations ===

    def remember(self, text: str, importance: float = 0.7, domain: str = "general"):
        """Explicitly store something in memory"""
        self.memory.store(
            text=text,
            memory_type="preference",
            importance=importance,
            domain=domain
        )
        print(f"[Clara] Stored: {text[:50]}...")

    def recall(self, query: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Query Clara's memory"""
        return self.memory.recall(query, top_k=top_k)

    def recall_by_entity(self, entity: str) -> List[Memory]:
        """Find memories by entity"""
        return self.memory.recall_by_entity(entity)

    def save_memory(self, path: str = None):
        """Save memory to disk"""
        path = path or os.path.join(self.models_dir, "..", self.config.memory_file)
        self.memory.save(path)

    def load_memory(self, path: str = None) -> bool:
        """Load memory from disk"""
        path = path or os.path.join(self.models_dir, "..", self.config.memory_file)
        return self.memory.load(path)

    # === Diagnostics ===

    def stats(self) -> Dict:
        """Get Clara's statistics"""
        mem_stats = self.memory.stats()
        return {
            "interactions": self.interaction_count,
            "memory": mem_stats,
            "router_mode": self.config.router_mode,
            "brains_loaded": {
                "knowledge": self._knowledge_model is not None,
                "personality": self._personality_model is not None,
            }
        }

    def debug_mode(self, enabled: bool = True):
        """Toggle debug mode"""
        self.config.debug = enabled
        if self._memory:
            self._memory.debug = enabled
        if hasattr(self._router, 'debug'):
            self._router.debug = enabled
        print(f"[Clara] Debug mode: {'ON' if enabled else 'OFF'}")

    # === Tool Registration (Future) ===

    def register_tool(self, name: str, description: str,
                      parameters: Dict, handler: callable):
        """Register a tool for Clara to use"""
        if isinstance(self.router, HybridRouter) and self.router.llm_router:
            self.router.llm_router.register_tool(name, description, parameters, handler)
        elif isinstance(self.router, NemotronRouter):
            self.router.register_tool(name, description, parameters, handler)
        else:
            print(f"[Clara] Warning: Router doesn't support tool registration")


# === Convenience Functions ===

def create_clara(models_dir: str = "./models",
                 router_mode: str = "embedding",
                 debug: bool = False,
                 **kwargs) -> ClaraV2:
    """
    Create a Clara instance with common configuration.

    Args:
        models_dir: Directory containing model files
        router_mode: "embedding" (fast), "llm" (smart), "hybrid" (balanced)
        debug: Enable debug output
        **kwargs: Additional config options

    Returns:
        ClaraV2 instance
    """
    config = ClaraConfig(
        router_mode=router_mode,
        debug=debug,
        **kwargs
    )
    return ClaraV2(config=config, models_dir=models_dir)


# === Demo ===

def demo_clara_v2():
    """Demonstrate Clara v2 capabilities"""
    print("=" * 70)
    print("CLARA v2 - DEMONSTRATION")
    print("=" * 70)

    # Create Clara with embedding router (no heavy models needed for demo)
    config = ClaraConfig(
        router_mode="embedding",
        debug=True,
        hdc_dim=64000
    )

    clara = ClaraV2(config=config)

    # Initialize memory and router (lightweight)
    print("\n1. Initializing components...")
    _ = clara.memory  # Trigger lazy load
    _ = clara.router  # Trigger lazy load

    # Store some memories
    print("\n2. Storing memories...")
    clara.remember("User likes Python programming", importance=0.8, domain="coding")
    clara.remember("User offered me coffee earlier", importance=0.6, domain="personality")
    clara.remember("We discussed async/await patterns", importance=0.7, domain="coding")

    # Test memory recall
    print("\n3. Testing memory recall...")
    results = clara.recall("Python async")
    for mem, score in results:
        print(f"   [{score:.3f}] {mem.text[:50]}...")

    # Test routing (without actual generation)
    print("\n4. Testing routing decisions...")
    test_queries = [
        "How do I use asyncio in Python?",
        "I'm feeling stressed today",
        "What is quantum entanglement?",
        "How's the coffee?",
    ]

    for query in test_queries:
        decision = clara.router.route(query)
        print(f"   '{query[:40]}...'")
        print(f"      → {decision.brain.value}/{decision.domain.value} ({decision.confidence:.2f})")

    # Stats
    print("\n5. Clara statistics:")
    stats = clara.stats()
    print(f"   Memory size: {stats['memory']['size_kb']:.1f} KB")
    print(f"   Total memories: {stats['memory']['total_memories']}")
    print(f"   HDC dimensions: {stats['memory']['dimensions']:,}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo use Clara with actual models:")
    print("   clara = create_clara(models_dir='/path/to/models')")
    print("   clara.load_brains()")
    print("   response = clara.chat('Hello!')")


if __name__ == "__main__":
    demo_clara_v2()
