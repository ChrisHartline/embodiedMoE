"""
Session Memory Layer - Fast in-memory cache with FalkorDB backend

This module provides:
- L1 session cache (Redis protocol via FalkorDB)
- Graph storage for entity relationships
- Vector search for semantic similarity
- Async-first design for non-blocking operations

FalkorDB provides Redis compatibility + Graph + Vector in one container.
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import hashlib


# Optional imports - graceful degradation
try:
    from falkordb import FalkorDB
    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    routing: Optional[Dict] = None  # Routing decision metadata
    entities: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Current session state"""
    session_id: str
    user_id: Optional[str] = None
    started: float = field(default_factory=time.time)
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: Dict[str, str] = field(default_factory=dict)  # slot: value
    routing_cache: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started": self.started,
            "turns": [asdict(t) for t in self.turns],
            "active_entities": self.active_entities,
            "routing_cache": self.routing_cache,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionState":
        turns = [ConversationTurn(**t) for t in data.get("turns", [])]
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            started=data.get("started", time.time()),
            turns=turns,
            active_entities=data.get("active_entities", {}),
            routing_cache=data.get("routing_cache", {}),
            metadata=data.get("metadata", {})
        )


class InMemorySessionStore:
    """
    Simple in-memory session store (fallback when no Redis/FalkorDB).
    Useful for testing and single-process deployments.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.sessions: Dict[str, SessionState] = {}
        self.ttl = ttl_seconds
        self._last_cleanup = time.time()

    def _cleanup_expired(self):
        """Remove expired sessions"""
        now = time.time()
        if now - self._last_cleanup < 60:  # Check every minute
            return

        cutoff = now - self.ttl
        expired = [sid for sid, s in self.sessions.items() if s.started < cutoff]
        for sid in expired:
            del self.sessions[sid]
        self._last_cleanup = now

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        self._cleanup_expired()
        return self.sessions.get(session_id)

    async def save_session(self, session: SessionState):
        self.sessions[session.session_id] = session

    async def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    async def add_turn(self, session_id: str, turn: ConversationTurn):
        session = self.sessions.get(session_id)
        if session:
            session.turns.append(turn)

    async def get_recent_turns(self, session_id: str, n: int = 5) -> List[ConversationTurn]:
        session = self.sessions.get(session_id)
        if session:
            return session.turns[-n:]
        return []

    async def set_entity(self, session_id: str, slot: str, value: str):
        session = self.sessions.get(session_id)
        if session:
            session.active_entities[slot] = value

    async def get_entities(self, session_id: str) -> Dict[str, str]:
        session = self.sessions.get(session_id)
        return session.active_entities if session else {}


class FalkorDBSessionStore:
    """
    FalkorDB-backed session store with graph capabilities.

    Uses Redis protocol for fast key-value access and
    Cypher queries for graph traversal.

    Default connection (ClaraLily container):
    - Host: localhost
    - Port: 32758 (mapped from container 6378)
    - Graph: clara
    """

    # Default connection for ClaraLily container
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 32758  # Docker mapping: 32758:6378
    DEFAULT_GRAPH = "clara"

    def __init__(self,
                 host: str = None,
                 port: int = None,
                 graph_name: str = None,
                 ttl_seconds: int = 3600,
                 debug: bool = False):
        """
        Initialize FalkorDB session store.

        Args:
            host: FalkorDB host (default: localhost)
            port: FalkorDB port (default: 32758 for ClaraLily container)
            graph_name: Name of the graph to use (default: clara)
            ttl_seconds: Session TTL
            debug: Enable debug output
        """
        if not HAS_FALKORDB:
            raise ImportError("falkordb package required. Install with: pip install falkordb")

        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.graph_name = graph_name or self.DEFAULT_GRAPH
        self.ttl = ttl_seconds
        self.debug = debug

        self._db: Optional[FalkorDB] = None
        self._graph = None

    def _connect(self):
        """Lazy connection to FalkorDB"""
        if self._db is None:
            self._db = FalkorDB(host=self.host, port=self.port)
            self._graph = self._db.select_graph(self.graph_name)
            self._ensure_schema()
            if self.debug:
                print(f"   [FalkorDB] Connected to {self.host}:{self.port}/{self.graph_name}")

    def _ensure_schema(self):
        """Create indexes if they don't exist"""
        try:
            # Create indexes for common queries
            self._graph.query("CREATE INDEX IF NOT EXISTS FOR (s:Session) ON (s.id)")
            self._graph.query("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            self._graph.query("CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)")
        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Schema setup: {e}")

    @property
    def graph(self):
        self._connect()
        return self._graph

    @property
    def db(self):
        self._connect()
        return self._db

    # === Session Operations (Key-Value) ===

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session from Redis key"""
        self._connect()
        try:
            # Use underlying Redis connection
            data = self._db._conn.get(f"session:{session_id}")
            if data:
                return SessionState.from_dict(json.loads(data))
        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Get session error: {e}")
        return None

    async def save_session(self, session: SessionState):
        """Save session to Redis key with TTL"""
        self._connect()
        try:
            key = f"session:{session.session_id}"
            data = json.dumps(session.to_dict())
            self._db._conn.setex(key, self.ttl, data)
        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Save session error: {e}")

    async def delete_session(self, session_id: str):
        """Delete session"""
        self._connect()
        try:
            self._db._conn.delete(f"session:{session_id}")
        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Delete session error: {e}")

    async def add_turn(self, session_id: str, turn: ConversationTurn):
        """Add turn to session"""
        session = await self.get_session(session_id)
        if session:
            session.turns.append(turn)
            await self.save_session(session)

    async def get_recent_turns(self, session_id: str, n: int = 5) -> List[ConversationTurn]:
        """Get last N turns"""
        session = await self.get_session(session_id)
        if session:
            return session.turns[-n:]
        return []

    async def set_entity(self, session_id: str, slot: str, value: str):
        """Set active entity slot"""
        session = await self.get_session(session_id)
        if session:
            session.active_entities[slot] = value
            await self.save_session(session)

    async def get_entities(self, session_id: str) -> Dict[str, str]:
        """Get active entities"""
        session = await self.get_session(session_id)
        return session.active_entities if session else {}

    # === Graph Operations ===

    async def store_memory_graph(self,
                                  session_id: str,
                                  text: str,
                                  domain: str,
                                  entities: List[str],
                                  importance: float = 0.5,
                                  embedding: Optional[List[float]] = None):
        """
        Store memory in graph with entity relationships.

        Creates:
        - Memory node with text, domain, timestamp
        - Entity nodes for each mentioned entity
        - MENTIONS relationships between Memory and Entity
        - PART_OF relationship to Session
        """
        self._connect()

        memory_id = hashlib.md5(f"{session_id}:{text}:{time.time()}".encode()).hexdigest()[:12]
        timestamp = time.time()

        try:
            # Create memory node
            params = {
                "mid": memory_id,
                "sid": session_id,
                "text": text[:500],  # Truncate for graph storage
                "domain": domain,
                "importance": importance,
                "timestamp": timestamp
            }

            # Add embedding if provided
            if embedding:
                params["embedding"] = embedding

            self._graph.query("""
                MERGE (s:Session {id: $sid})
                CREATE (m:Memory {
                    id: $mid,
                    text: $text,
                    domain: $domain,
                    importance: $importance,
                    timestamp: $timestamp
                })
                CREATE (s)-[:CONTAINS]->(m)
            """, params)

            # Create entity nodes and relationships
            for entity in entities:
                self._graph.query("""
                    MATCH (m:Memory {id: $mid})
                    MERGE (e:Entity {name: $entity})
                    ON CREATE SET e.first_seen = $ts
                    SET e.last_seen = $ts
                    MERGE (m)-[:MENTIONS]->(e)
                """, {"mid": memory_id, "entity": entity.upper(), "ts": timestamp})

            if self.debug:
                print(f"   [FalkorDB] Stored memory {memory_id} with {len(entities)} entities")

        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Store memory error: {e}")

    async def find_entity_memories(self,
                                    entity: str,
                                    limit: int = 5,
                                    since: Optional[float] = None) -> List[Dict]:
        """Find memories mentioning an entity"""
        self._connect()

        try:
            params = {"entity": entity.upper(), "limit": limit}
            query = """
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $entity})
            """

            if since:
                query += " WHERE m.timestamp > $since"
                params["since"] = since

            query += """
                RETURN m.id, m.text, m.domain, m.timestamp, m.importance
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """

            result = self._graph.query(query, params)

            memories = []
            for row in result.result_set:
                memories.append({
                    "id": row[0],
                    "text": row[1],
                    "domain": row[2],
                    "timestamp": row[3],
                    "importance": row[4]
                })

            return memories

        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Find entity memories error: {e}")
            return []

    async def find_related_entities(self, entity: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Find entities that co-occur with given entity"""
        self._connect()

        try:
            result = self._graph.query("""
                MATCH (e1:Entity {name: $entity})<-[:MENTIONS]-(m:Memory)-[:MENTIONS]->(e2:Entity)
                WHERE e1 <> e2
                RETURN e2.name, count(m) as co_occurrences
                ORDER BY co_occurrences DESC
                LIMIT $limit
            """, {"entity": entity.upper(), "limit": limit})

            return [(row[0], row[1]) for row in result.result_set]

        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Find related entities error: {e}")
            return []

    async def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of session from graph"""
        self._connect()

        try:
            result = self._graph.query("""
                MATCH (s:Session {id: $sid})-[:CONTAINS]->(m:Memory)
                OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
                RETURN count(DISTINCT m) as memory_count,
                       collect(DISTINCT e.name) as entities,
                       collect(DISTINCT m.domain) as domains
            """, {"sid": session_id})

            if result.result_set:
                row = result.result_set[0]
                return {
                    "memory_count": row[0],
                    "entities": row[1] or [],
                    "domains": row[2] or []
                }

        except Exception as e:
            if self.debug:
                print(f"   [FalkorDB] Session summary error: {e}")

        return {"memory_count": 0, "entities": [], "domains": []}


class SessionMemory:
    """
    High-level session memory interface.

    Automatically selects backend:
    1. FalkorDB if available and configured
    2. Redis if available
    3. In-memory fallback

    Default connection (ClaraLily container):
    - Host: localhost
    - Port: 32758
    - Graph: clara
    """

    def __init__(self,
                 backend: str = "auto",
                 host: str = None,
                 port: int = None,
                 graph_name: str = None,
                 ttl_seconds: int = 3600,
                 max_turns: int = 20,
                 debug: bool = False):
        """
        Initialize session memory.

        Args:
            backend: "falkordb", "redis", "memory", or "auto"
            host: Database host (default: localhost)
            port: Database port (default: 32758 for ClaraLily)
            graph_name: FalkorDB graph name (default: clara)
            ttl_seconds: Session TTL
            max_turns: Max turns to keep in session
            debug: Enable debug output
        """
        self.max_turns = max_turns
        self.debug = debug

        # Use FalkorDB defaults
        host = host or FalkorDBSessionStore.DEFAULT_HOST
        port = port or FalkorDBSessionStore.DEFAULT_PORT
        graph_name = graph_name or FalkorDBSessionStore.DEFAULT_GRAPH

        # Select backend
        if backend == "auto":
            if HAS_FALKORDB:
                backend = "falkordb"
            elif HAS_REDIS:
                backend = "redis"
            else:
                backend = "memory"

        if backend == "falkordb":
            self.store = FalkorDBSessionStore(
                host=host,
                port=port,
                graph_name=graph_name,
                ttl_seconds=ttl_seconds,
                debug=debug
            )
            self.has_graph = True
            print(f"   SessionMemory initialized (FalkorDB: {host}:{port}/{graph_name})")
        elif backend == "redis":
            # TODO: Implement Redis-only store
            self.store = InMemorySessionStore(ttl_seconds=ttl_seconds)
            self.has_graph = False
            print(f"   SessionMemory initialized (in-memory fallback, Redis TODO)")
        else:
            self.store = InMemorySessionStore(ttl_seconds=ttl_seconds)
            self.has_graph = False
            print(f"   SessionMemory initialized (in-memory)")

        self.backend = backend

    async def start_session(self,
                            session_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> SessionState:
        """Start a new session or resume existing"""
        if session_id:
            existing = await self.store.get_session(session_id)
            if existing:
                if self.debug:
                    print(f"   [Session] Resumed session {session_id}")
                return existing

        # Generate new session ID
        if not session_id:
            session_id = hashlib.md5(f"{time.time()}:{user_id}".encode()).hexdigest()[:12]

        session = SessionState(session_id=session_id, user_id=user_id)
        await self.store.save_session(session)

        if self.debug:
            print(f"   [Session] Started new session {session_id}")

        return session

    async def add_turn(self,
                       session_id: str,
                       role: str,
                       content: str,
                       routing: Optional[Dict] = None,
                       entities: Optional[List[str]] = None) -> ConversationTurn:
        """Add a conversation turn"""
        turn = ConversationTurn(
            role=role,
            content=content,
            routing=routing,
            entities=entities or []
        )

        await self.store.add_turn(session_id, turn)

        # Store in graph if available
        if self.has_graph and entities:
            domain = routing.get("domain", "general") if routing else "general"
            await self.store.store_memory_graph(
                session_id=session_id,
                text=content,
                domain=domain,
                entities=entities
            )

        return turn

    async def get_context(self, session_id: str, n_turns: int = 5) -> str:
        """Get recent context as string for prompts"""
        turns = await self.store.get_recent_turns(session_id, n_turns)

        if not turns:
            return ""

        parts = ["[Recent conversation:]"]
        for turn in turns:
            role = "User" if turn.role == "user" else "Clara"
            parts.append(f"- {role}: {turn.content[:100]}")

        return "\n".join(parts)

    async def get_active_entities(self, session_id: str) -> Dict[str, str]:
        """Get currently active entity slots"""
        return await self.store.get_entities(session_id)

    async def set_entity(self, session_id: str, slot: str, value: str):
        """Set an entity slot"""
        await self.store.set_entity(session_id, slot, value)

    async def find_by_entity(self, entity: str, limit: int = 5) -> List[Dict]:
        """Find memories mentioning an entity (graph backend only)"""
        if self.has_graph:
            return await self.store.find_entity_memories(entity, limit)
        return []

    async def find_related(self, entity: str) -> List[Tuple[str, int]]:
        """Find entities related to given entity (graph backend only)"""
        if self.has_graph:
            return await self.store.find_related_entities(entity)
        return []


# === Sync Wrapper for non-async usage ===

class SyncSessionMemory:
    """Synchronous wrapper for SessionMemory"""

    def __init__(self, **kwargs):
        self._async_mem = SessionMemory(**kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        return self._get_loop().run_until_complete(coro)

    def start_session(self, session_id=None, user_id=None):
        return self._run(self._async_mem.start_session(session_id, user_id))

    def add_turn(self, session_id, role, content, routing=None, entities=None):
        return self._run(self._async_mem.add_turn(session_id, role, content, routing, entities))

    def get_context(self, session_id, n_turns=5):
        return self._run(self._async_mem.get_context(session_id, n_turns))

    def get_active_entities(self, session_id):
        return self._run(self._async_mem.get_active_entities(session_id))

    def set_entity(self, session_id, slot, value):
        return self._run(self._async_mem.set_entity(session_id, slot, value))

    def find_by_entity(self, entity, limit=5):
        return self._run(self._async_mem.find_by_entity(entity, limit))

    def find_related(self, entity):
        return self._run(self._async_mem.find_related(entity))


# === Demo ===

def demo_session_memory():
    """Demonstrate session memory (in-memory fallback)"""
    print("=" * 60)
    print("SESSION MEMORY - DEMONSTRATION")
    print("=" * 60)

    # Use sync wrapper for demo
    mem = SyncSessionMemory(backend="memory", debug=True)

    # Start session
    print("\n1. Starting session...")
    session = mem.start_session(user_id="demo_user")
    sid = session.session_id
    print(f"   Session ID: {sid}")

    # Add turns
    print("\n2. Adding conversation turns...")
    mem.add_turn(sid, "user", "Hi! Would you like some coffee?", entities=["COFFEE"])
    mem.add_turn(sid, "assistant", "That sounds wonderful, thank you!")
    mem.add_turn(sid, "user", "How do I fix a Python error?",
                 routing={"domain": "coding"}, entities=["PYTHON"])
    mem.add_turn(sid, "assistant", "I'd be happy to help with Python!")

    # Get context
    print("\n3. Getting context...")
    context = mem.get_context(sid, n_turns=3)
    print(context)

    # Set entities
    print("\n4. Setting entity slots...")
    mem.set_entity(sid, "DRINK", "coffee")
    mem.set_entity(sid, "TOPIC", "python")

    entities = mem.get_active_entities(sid)
    print(f"   Active entities: {entities}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_session_memory()
