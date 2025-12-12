"""
Nemotron-based Router for Clara

A sophisticated routing system using NVIDIA Nemotron models (or compatible alternatives)
for intent classification, domain routing, and future tool-calling capabilities.

Supported models:
- nvidia/Llama-3.1-Nemotron-Nano-8B-v1 (recommended)
- nvidia/NVIDIA-Nemotron-Nano-9B-v2
- Qwen/Qwen2.5-1.5B-Instruct (lightweight alternative)
- Any instruction-tuned model compatible with chat templates

Features:
- Structured JSON output for routing decisions
- Multi-intent parsing
- Confidence scoring
- Tool-calling scaffold for future expansion
- Memory-aware routing context
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch


class RoutingDomain(Enum):
    """Available routing domains"""
    MEDICAL = "medical"
    CODING = "coding"
    TEACHING = "teaching"
    QUANTUM = "quantum"
    PERSONALITY = "personality"
    GENERAL = "general"
    TOOL_CALL = "tool_call"  # For future tool-calling


class BrainType(Enum):
    """Brain types for routing"""
    KNOWLEDGE = "knowledge"
    PERSONALITY = "personality"
    TOOL = "tool"  # For future tool execution


@dataclass
class RoutingDecision:
    """Structured routing decision from the router"""
    brain: BrainType
    domain: RoutingDomain
    confidence: float
    sub_intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    requires_memory: bool = False
    reasoning: Optional[str] = None
    raw_output: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "brain": self.brain.value,
            "domain": self.domain.value,
            "confidence": self.confidence,
            "sub_intent": self.sub_intent,
            "entities": self.entities,
            "requires_memory": self.requires_memory,
            "reasoning": self.reasoning
        }

    @classmethod
    def default_personality(cls) -> "RoutingDecision":
        """Default fallback to personality routing"""
        return cls(
            brain=BrainType.PERSONALITY,
            domain=RoutingDomain.PERSONALITY,
            confidence=0.5,
            sub_intent="general_conversation"
        )


@dataclass
class ToolCall:
    """Represents a tool call request (for future use)"""
    tool_name: str
    arguments: Dict[str, Any]
    confidence: float


class NemotronRouter:
    """
    LLM-based router using NVIDIA Nemotron-Orchestrator-8B.

    Nemotron-Orchestrator-8B is specifically trained for:
    - Model and tool orchestration via reinforcement learning (GRPO)
    - Multi-turn agentic task coordination
    - Structured JSON tool calls
    - Efficient routing decisions

    Capabilities:
    - Intent classification with chain-of-thought reasoning
    - Multi-domain routing to specialized brains
    - Structured JSON output for routing decisions
    - Native tool-calling support
    - Memory-aware context handling
    """

    # Default model - Nemotron-Orchestrator-8B is purpose-built for routing
    DEFAULT_MODEL = "nvidia/Nemotron-Orchestrator-8B"

    # Routing prompt template (aligned with Orchestrator training format)
    ROUTING_PROMPT = """You are Clara's orchestrator, routing queries to specialized AI brains.

## Available Brains/Tools:
- knowledge_medical: Medical knowledge, symptoms, diagnosis, treatment
- knowledge_coding: Programming, software, debugging, APIs, algorithms
- knowledge_teaching: Explanations, tutorials, fundamentals, learning
- knowledge_quantum: Quantum physics, quantum computing, entanglement
- personality_warmth: Emotional support, casual chat, greetings, encouragement

## Task:
Analyze the user message and route to the best brain. Output a JSON tool call.

{memory_context}

User message: {query}

Think step by step, then output your routing decision as JSON:
{{
    "tool_call": {{
        "name": "<brain_name>",
        "arguments": {{
            "query": "<user query>",
            "confidence": <0.0-1.0>,
            "sub_intent": "<specific intent>",
            "entities": [<key entities>],
            "requires_memory": <true/false>
        }}
    }},
    "reasoning": "<chain of thought>"
}}

Response:"""

    # Tool-calling prompt (native Orchestrator format)
    TOOL_PROMPT = """You are an orchestrator that routes to tools and models.

## Available Tools:
{tool_descriptions}

## Task:
Analyze if any tools are needed for the user's request.

User message: {query}

Think step by step about which tool to use, then output:
{{
    "tool_call": {{
        "name": "<tool_name>",
        "arguments": {{...}}
    }},
    "reasoning": "<your reasoning>"
}}

Or if no tool is needed:
{{
    "tool_call": null,
    "response_needed": true,
    "reasoning": "<why no tool needed>"
}}

Response:"""

    def __init__(self,
                 model_name: str = None,
                 device: str = "auto",
                 load_in_8bit: bool = True,
                 use_flash_attention: bool = True,
                 debug: bool = False):
        """
        Initialize the Nemotron Orchestrator router.

        Args:
            model_name: HuggingFace model identifier (default: Nemotron-Orchestrator-8B)
            device: Device for inference ('auto', 'cuda', 'cpu')
            load_in_8bit: Use 8-bit quantization for memory efficiency
            use_flash_attention: Use flash attention if available
            debug: Enable debug output
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.debug = debug
        self.model = None
        self.tokenizer = None
        self._loaded = False

        # Tool registry (for future tool-calling)
        self.tools: Dict[str, Dict] = {}

        # Store config for lazy loading
        self._load_config = {
            "load_in_8bit": load_in_8bit,
            "use_flash_attention": use_flash_attention
        }

        print(f"   NemotronRouter initialized (model: {model_name})")
        print(f"   Call .load() to load the model into memory")

    def load(self):
        """Load the model into memory (lazy loading)"""
        if self._loaded:
            print("   Model already loaded")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"   Loading router model: {self.model_name}")

        # Configure quantization
        quantization_config = None
        if self._load_config["load_in_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device,
            "trust_remote_code": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self._load_config["use_flash_attention"]:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"   Warning: Failed with flash attention, retrying without: {e}")
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

        self.model.eval()
        self._loaded = True

        # Get model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"   Router model loaded: {param_count / 1e9:.1f}B parameters")

    def _ensure_loaded(self):
        """Ensure model is loaded before inference"""
        if not self._loaded:
            self.load()

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text from prompt"""
        self._ensure_loaded()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temp for consistent routing
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from model response (handles nested JSON from Orchestrator)"""
        # Try to find the outermost JSON object (may contain nested objects)
        # Use a more robust regex for nested structures
        brace_count = 0
        start_idx = None

        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = response[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try next match
                        start_idx = None
                        continue

        # Fallback: simple regex for flat JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to parse entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        if self.debug:
            print(f"      [ROUTER] Failed to parse JSON: {response[:100]}...")

        return None

    def route(self,
              query: str,
              memory_context: Optional[str] = None,
              threshold: float = 0.3) -> RoutingDecision:
        """
        Route a query to the appropriate brain/domain.

        Args:
            query: User's input message
            memory_context: Optional conversation context from memory
            threshold: Minimum confidence threshold

        Returns:
            RoutingDecision with brain, domain, confidence, and metadata
        """
        # Build prompt
        memory_str = ""
        if memory_context:
            memory_str = f"\nConversation context:\n{memory_context}\n"

        prompt = self.ROUTING_PROMPT.format(
            memory_context=memory_str,
            query=query
        )

        if self.debug:
            print(f"      [ROUTER] Query: '{query[:50]}...'")

        # Generate response
        response = self._generate(prompt, max_new_tokens=150)

        if self.debug:
            print(f"      [ROUTER] Response: {response[:100]}...")

        # Parse JSON
        parsed = self._parse_json_response(response)

        if parsed is None:
            # Fallback to keyword-based routing
            return self._fallback_route(query)

        # Build RoutingDecision
        try:
            # Handle Orchestrator-style tool_call format
            if "tool_call" in parsed and parsed["tool_call"]:
                tool_call = parsed["tool_call"]
                tool_name = tool_call.get("name", "personality_warmth")
                arguments = tool_call.get("arguments", {})

                # Parse brain_domain from tool name (e.g., "knowledge_coding")
                brain, domain = self._parse_tool_name(tool_name)
                confidence = float(arguments.get("confidence", 0.7))
                sub_intent = arguments.get("sub_intent")
                entities = arguments.get("entities", [])
                requires_memory = arguments.get("requires_memory", False)
                reasoning = parsed.get("reasoning")

            else:
                # Legacy format: direct domain/confidence
                domain_str = parsed.get("domain", "personality").lower()
                domain = RoutingDomain(domain_str) if domain_str in [d.value for d in RoutingDomain] else RoutingDomain.PERSONALITY

                # Determine brain type
                if domain in [RoutingDomain.MEDICAL, RoutingDomain.CODING,
                             RoutingDomain.TEACHING, RoutingDomain.QUANTUM]:
                    brain = BrainType.KNOWLEDGE
                elif domain == RoutingDomain.TOOL_CALL:
                    brain = BrainType.TOOL
                else:
                    brain = BrainType.PERSONALITY

                confidence = float(parsed.get("confidence", 0.5))
                sub_intent = parsed.get("sub_intent")
                entities = parsed.get("entities", [])
                requires_memory = parsed.get("requires_memory", False)
                reasoning = parsed.get("reasoning")

            # Apply threshold
            if confidence < threshold and brain == BrainType.KNOWLEDGE:
                brain = BrainType.PERSONALITY
                domain = RoutingDomain.PERSONALITY
                confidence = max(confidence, 0.4)

            decision = RoutingDecision(
                brain=brain,
                domain=domain,
                confidence=confidence,
                sub_intent=sub_intent,
                entities=entities if isinstance(entities, list) else [],
                requires_memory=requires_memory,
                reasoning=reasoning,
                raw_output=response
            )

            if self.debug:
                print(f"      [ROUTER] Decision: {brain.value}/{domain.value} ({confidence:.2f})")

            return decision

        except Exception as e:
            if self.debug:
                print(f"      [ROUTER] Parse error: {e}")
            return self._fallback_route(query)

    def _parse_tool_name(self, tool_name: str) -> Tuple[BrainType, RoutingDomain]:
        """Parse Orchestrator tool name into brain type and domain"""
        tool_name = tool_name.lower()

        # Map tool names to brain/domain
        mappings = {
            "knowledge_medical": (BrainType.KNOWLEDGE, RoutingDomain.MEDICAL),
            "knowledge_coding": (BrainType.KNOWLEDGE, RoutingDomain.CODING),
            "knowledge_teaching": (BrainType.KNOWLEDGE, RoutingDomain.TEACHING),
            "knowledge_quantum": (BrainType.KNOWLEDGE, RoutingDomain.QUANTUM),
            "personality_warmth": (BrainType.PERSONALITY, RoutingDomain.PERSONALITY),
            "personality_playful": (BrainType.PERSONALITY, RoutingDomain.PERSONALITY),
            "personality_encouragement": (BrainType.PERSONALITY, RoutingDomain.PERSONALITY),
        }

        if tool_name in mappings:
            return mappings[tool_name]

        # Try to parse generic format: brain_domain
        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            brain_str, domain_str = parts[0], parts[1]

            if brain_str == "knowledge":
                brain = BrainType.KNOWLEDGE
            elif brain_str == "tool":
                brain = BrainType.TOOL
            else:
                brain = BrainType.PERSONALITY

            try:
                domain = RoutingDomain(domain_str)
            except ValueError:
                domain = RoutingDomain.PERSONALITY

            return brain, domain

        # Default fallback
        return BrainType.PERSONALITY, RoutingDomain.PERSONALITY

    def _fallback_route(self, query: str) -> RoutingDecision:
        """Keyword-based fallback routing when LLM parsing fails"""
        query_lower = query.lower()

        # Keyword patterns for each domain
        patterns = {
            RoutingDomain.CODING: [
                "code", "python", "javascript", "function", "error", "bug",
                "debug", "api", "database", "programming", "software", "git"
            ],
            RoutingDomain.MEDICAL: [
                "symptom", "pain", "doctor", "medicine", "health", "sick",
                "diagnosis", "treatment", "headache", "fever", "illness"
            ],
            RoutingDomain.TEACHING: [
                "explain", "how does", "what is", "teach", "learn", "tutorial",
                "understand", "basics", "fundamentals", "example"
            ],
            RoutingDomain.QUANTUM: [
                "quantum", "qubit", "superposition", "entanglement",
                "wave function", "schrodinger", "particle"
            ],
        }

        # Score each domain
        scores = {domain: 0 for domain in patterns}

        for domain, keywords in patterns.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[domain] += 1

        # Find best match
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score > 0:
            confidence = min(0.3 + (best_score * 0.15), 0.8)
            brain = BrainType.KNOWLEDGE
            domain = best_domain
        else:
            confidence = 0.6
            brain = BrainType.PERSONALITY
            domain = RoutingDomain.PERSONALITY

        return RoutingDecision(
            brain=brain,
            domain=domain,
            confidence=confidence,
            sub_intent="fallback_keyword_match",
            reasoning="Fallback routing via keyword matching"
        )

    # === Tool Calling (Future Capability) ===

    def register_tool(self, name: str, description: str,
                      parameters: Dict[str, Any],
                      handler: Optional[callable] = None):
        """
        Register a tool for potential tool-calling.

        Args:
            name: Tool name
            description: What the tool does
            parameters: JSON schema of parameters
            handler: Optional callable to execute the tool
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        if self.debug:
            print(f"      [ROUTER] Registered tool: {name}")

    def check_tool_call(self, query: str) -> Optional[ToolCall]:
        """
        Check if query requires a tool call.

        Args:
            query: User's message

        Returns:
            ToolCall if tool is needed, None otherwise
        """
        if not self.tools:
            return None

        # Build tool descriptions
        tool_desc_parts = []
        for name, tool in self.tools.items():
            params_str = json.dumps(tool["parameters"], indent=2)
            tool_desc_parts.append(
                f"- {name}: {tool['description']}\n  Parameters: {params_str}"
            )
        tool_descriptions = "\n".join(tool_desc_parts)

        prompt = self.TOOL_PROMPT.format(
            tool_descriptions=tool_descriptions,
            query=query
        )

        response = self._generate(prompt, max_new_tokens=200)
        parsed = self._parse_json_response(response)

        if parsed and parsed.get("tool_call"):
            tc = parsed["tool_call"]
            return ToolCall(
                tool_name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
                confidence=0.8  # TODO: Get from model
            )

        return None

    def execute_tool(self, tool_call: ToolCall) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result
        """
        if tool_call.tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_call.tool_name}")

        tool = self.tools[tool_call.tool_name]
        handler = tool.get("handler")

        if handler is None:
            raise ValueError(f"No handler for tool: {tool_call.tool_name}")

        return handler(**tool_call.arguments)

    # === Batch Routing ===

    def route_batch(self, queries: List[str],
                    memory_contexts: Optional[List[str]] = None) -> List[RoutingDecision]:
        """
        Route multiple queries (useful for preprocessing).

        Args:
            queries: List of user messages
            memory_contexts: Optional list of memory contexts

        Returns:
            List of RoutingDecisions
        """
        if memory_contexts is None:
            memory_contexts = [None] * len(queries)

        return [
            self.route(q, mc)
            for q, mc in zip(queries, memory_contexts)
        ]


class EmbeddingRouter:
    """
    Fast embedding-based router (no LLM inference).

    Uses sentence embeddings for similarity-based routing.
    Much faster than LLM routing but less nuanced.
    Good for latency-critical applications or as a pre-filter.
    """

    def __init__(self, embedder=None, debug: bool = False):
        """
        Initialize embedding router.

        Args:
            embedder: Optional sentence transformer (loads default if None)
            debug: Enable debug output
        """
        self.debug = debug

        if embedder is not None:
            self.embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Domain descriptions for embedding comparison
        self.domain_descriptions = {
            RoutingDomain.MEDICAL: """
                symptoms diagnosis treatment disease illness pain fever infection
                headache nauseous dizzy blood pressure heart lungs brain body
                doctor hospital medicine medication prescription surgery vaccine
            """,
            RoutingDomain.CODING: """
                programming code software python javascript java function method
                variable array list loop error exception bug debug API database
                algorithm class object compile runtime syntax git repository
            """,
            RoutingDomain.TEACHING: """
                explain how does work basics fundamentals introduction tutorial
                step by step concept theory lesson learn teach education student
                example analogy walk through guide overview what is
            """,
            RoutingDomain.QUANTUM: """
                quantum physics qubit superposition entanglement wave function
                particle measurement collapse observer quantum computer gate
                coherence decoherence probability amplitude interference
            """,
            RoutingDomain.PERSONALITY: """
                feeling emotion mood happy sad angry anxious worried stressed
                excited nervous scared lonely depressed overwhelmed frustrated
                relationship friend family love support talk vent chat greeting
            """,
        }

        # Pre-compute domain embeddings
        print("   Computing domain embeddings...")
        self.domain_embeddings = {}
        for domain, desc in self.domain_descriptions.items():
            clean_desc = " ".join(desc.split())
            self.domain_embeddings[domain] = self.embedder.encode(clean_desc)

        print("   EmbeddingRouter ready")

    def route(self, query: str,
              memory_context: Optional[str] = None,
              threshold: float = 0.20) -> RoutingDecision:
        """
        Route using embedding similarity.

        Args:
            query: User message
            memory_context: Unused (for API compatibility)
            threshold: Minimum similarity threshold

        Returns:
            RoutingDecision
        """
        import numpy as np

        query_emb = self.embedder.encode(query)

        # Compute similarities
        similarities = {}
        for domain, domain_emb in self.domain_embeddings.items():
            sim = np.dot(query_emb, domain_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(domain_emb)
            )
            similarities[domain] = float(sim)

        # Find best match
        sorted_domains = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_conf = sorted_domains[0]
        second_domain, second_conf = sorted_domains[1]

        # If close, prefer personality
        if best_conf - second_conf < 0.05 and second_domain == RoutingDomain.PERSONALITY:
            best_domain = RoutingDomain.PERSONALITY
            best_conf = second_conf

        # Low confidence fallback
        if best_conf < threshold and best_domain != RoutingDomain.PERSONALITY:
            if similarities[RoutingDomain.PERSONALITY] > 0.15:
                best_domain = RoutingDomain.PERSONALITY
                best_conf = similarities[RoutingDomain.PERSONALITY]

        # Determine brain type
        if best_domain in [RoutingDomain.MEDICAL, RoutingDomain.CODING,
                          RoutingDomain.TEACHING, RoutingDomain.QUANTUM]:
            brain = BrainType.KNOWLEDGE
        else:
            brain = BrainType.PERSONALITY

        if self.debug:
            print(f"      [EMB-ROUTER] {brain.value}/{best_domain.value} ({best_conf:.2f})")

        return RoutingDecision(
            brain=brain,
            domain=best_domain,
            confidence=best_conf,
            sub_intent="embedding_similarity",
            reasoning=f"Top domains: {sorted_domains[:3]}"
        )


class HybridRouter:
    """
    Hybrid router combining fast embedding pre-filter with LLM routing.

    Strategy:
    1. Fast embedding check for high-confidence routing
    2. Fall back to LLM for ambiguous cases
    3. Memory-aware adjustments
    """

    def __init__(self,
                 llm_router: Optional[NemotronRouter] = None,
                 embedding_router: Optional[EmbeddingRouter] = None,
                 llm_threshold: float = 0.6,
                 debug: bool = False):
        """
        Initialize hybrid router.

        Args:
            llm_router: LLM-based router (created if None)
            embedding_router: Embedding-based router (created if None)
            llm_threshold: Embedding confidence below which to use LLM
            debug: Enable debug output
        """
        self.debug = debug
        self.llm_threshold = llm_threshold

        self.embedding_router = embedding_router or EmbeddingRouter(debug=debug)
        self.llm_router = llm_router  # Lazy load LLM only when needed

        self._llm_model_name = NemotronRouter.DEFAULT_MODEL

        print("   HybridRouter ready (LLM will load on demand)")

    def route(self,
              query: str,
              memory_context: Optional[str] = None,
              force_llm: bool = False) -> RoutingDecision:
        """
        Route with hybrid strategy.

        Args:
            query: User message
            memory_context: Conversation context
            force_llm: Force LLM routing regardless of confidence

        Returns:
            RoutingDecision
        """
        # Fast embedding check first
        emb_decision = self.embedding_router.route(query)

        if self.debug:
            print(f"      [HYBRID] Embedding: {emb_decision.domain.value} ({emb_decision.confidence:.2f})")

        # Use embedding result if confident and not forced to LLM
        if not force_llm and emb_decision.confidence >= self.llm_threshold:
            return emb_decision

        # Fall back to LLM for ambiguous cases
        if self.llm_router is None:
            self.llm_router = NemotronRouter(
                model_name=self._llm_model_name,
                debug=self.debug
            )

        llm_decision = self.llm_router.route(query, memory_context)

        if self.debug:
            print(f"      [HYBRID] LLM: {llm_decision.domain.value} ({llm_decision.confidence:.2f})")

        # Merge decisions (prefer LLM when available)
        if llm_decision.confidence > emb_decision.confidence:
            return llm_decision
        else:
            return emb_decision


# === Convenience Functions ===

def create_router(mode: str = "hybrid",
                  model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                  debug: bool = False):
    """
    Create a router with the specified mode.

    Args:
        mode: "llm", "embedding", or "hybrid"
        model_name: Model for LLM routing
        debug: Enable debug output

    Returns:
        Router instance
    """
    if mode == "llm":
        return NemotronRouter(model_name=model_name, debug=debug)
    elif mode == "embedding":
        return EmbeddingRouter(debug=debug)
    elif mode == "hybrid":
        return HybridRouter(debug=debug)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# === Demo ===

def demo_router():
    """Demonstrate the routing system"""
    print("=" * 70)
    print("NEMOTRON ROUTER - DEMONSTRATION")
    print("=" * 70)

    # Use embedding router for demo (no LLM loading required)
    print("\n1. Creating embedding router...")
    router = EmbeddingRouter(debug=True)

    # Test queries
    test_queries = [
        "How do I fix a TypeError in Python?",
        "I'm feeling really stressed about my job",
        "What is quantum entanglement?",
        "I have a headache and fever",
        "Can you explain how neural networks work?",
        "Hi! How are you today?",
        "How's the coffee?",
    ]

    print("\n2. Testing routing decisions...")
    print("-" * 70)

    for query in test_queries:
        decision = router.route(query)
        print(f"\nQuery: '{query}'")
        print(f"   → {decision.brain.value}/{decision.domain.value} "
              f"(conf: {decision.confidence:.2f})")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_router()
