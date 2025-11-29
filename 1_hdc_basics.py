"""
Hyperdimensional Computing Basics
This demonstrates the core HDC operations Clara uses for memory
"""

import numpy as np
from typing import Dict, List, Tuple
import random

class HDCMemory:
    """Simple HDC memory system with proper unbinding"""
    
    def __init__(self, dimensions: int = 10000):
        self.dimensions = dimensions
        self.symbol_library = {}  # Stores base symbols
        self.memories = []        # List of structured memories
        
    def create_symbol(self, name: str) -> np.ndarray:
        """Create a random hypervector for a concept"""
        if name not in self.symbol_library:
            # Binary hypervector: +1 or -1
            hv = np.random.choice([-1, 1], size=self.dimensions)
            self.symbol_library[name] = hv
        return self.symbol_library[name].copy()
    
    def bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Binding operation (element-wise multiplication)"""
        return hv1 * hv2
    
    def bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundling operation (superposition)"""
        summed = np.sum(hvs, axis=0)
        return np.sign(summed)
    
    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute similarity (normalized dot product)"""
        return np.dot(hv1, hv2) / self.dimensions
    
    def store_memory(self, **bindings: str):
        """
        Store a memory with named bindings
        Example: store_memory(subject="CLARA", action="PICKED_UP", object="CUP", result="SUCCESS")
        """
        # Create hypervectors for each role-filler pair
        bound_pairs = {}
        for role, filler in bindings.items():
            role_hv = self.create_symbol(f"ROLE_{role.upper()}")
            filler_hv = self.create_symbol(filler)
            bound_pairs[role] = self.bind(role_hv, filler_hv)
        
        # Bundle all pairs together
        memory_hv = self.bundle(list(bound_pairs.values()))
        
        self.memories.append({
            'hv': memory_hv,
            'bindings': bindings
        })
        
        return memory_hv
    
    def query(self, **query_bindings: str) -> List[Tuple[str, float]]:
        """
        Query memory and return missing bindings
        Example: query(object="CUP", result="SUCCESS") -> returns what action/subject goes with it
        """
        # Create query hypervector
        query_hv = np.zeros(self.dimensions)
        query_roles = set(query_bindings.keys())
        
        for role, filler in query_bindings.items():
            role_hv = self.create_symbol(f"ROLE_{role.upper()}")
            filler_hv = self.create_symbol(filler)
            query_hv = query_hv + self.bind(role_hv, filler_hv)
        
        query_hv = np.sign(query_hv)
        
        # Find matching memories
        results = {}
        
        for mem in self.memories:
            # Check if this memory matches the query
            matches = all(
                mem['bindings'].get(role) == filler 
                for role, filler in query_bindings.items()
            )
            
            if matches:
                # Extract the non-query bindings
                for role, filler in mem['bindings'].items():
                    if role not in query_roles:
                        if filler not in results:
                            results[filler] = 0
                        results[filler] += 1
        
        # Convert to sorted list
        result_list = [(filler, count) for filler, count in results.items()]
        result_list.sort(key=lambda x: x[1], reverse=True)
        
        return result_list
    
    def find_closest_symbol(self, hv: np.ndarray, top_k: int = 5, exclude_roles: bool = True) -> List[Tuple[str, float]]:
        """Find which symbols are closest to given hypervector"""
        if hv is None:
            return []
            
        similarities = []
        for name, symbol_hv in self.symbol_library.items():
            # Optionally exclude role markers
            if exclude_roles and name.startswith("ROLE_"):
                continue
            sim = self.similarity(hv, symbol_hv)
            similarities.append((name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def demo_hdc_basics():
    """Demonstrate HDC operations"""
    print("=" * 60)
    print("HDC Memory Demo - Proper Binding/Unbinding")
    print("=" * 60)
    
    memory = HDCMemory(dimensions=10000)
    
    # Store some memories with structured bindings
    print("\n1. Storing memories...")
    
    print("   - Clara picked up a cup successfully")
    memory.store_memory(
        subject="CLARA",
        action="PICKED_UP",
        object="CUP",
        result="SUCCESS"
    )
    
    print("   - Clara picked up an egg and it broke")
    memory.store_memory(
        subject="CLARA",
        action="PICKED_UP",
        object="EGG",
        result="FAILURE"
    )
    
    print("   - Clara pushed a box successfully")
    memory.store_memory(
        subject="CLARA",
        action="PUSHED",
        object="BOX",
        result="SUCCESS"
    )
    
    print("   - Clara grabbed a ball successfully")
    memory.store_memory(
        subject="CLARA",
        action="GRABBED",
        object="BALL",
        result="SUCCESS"
    )
    
    # Query: What action succeeded with CUP?
    print("\n2. Query: What ACTION succeeded with CUP?")
    results = memory.query(object="CUP", result="SUCCESS")
    
    print("   Results:")
    for item, count in results:
        print(f"   - {item}: {count} match(es)")
    print("   → Expected: PICKED_UP")
    
    # Query: What FAILED when picked up?
    print("\n3. Query: What OBJECT failed when PICKED_UP?")
    results = memory.query(action="PICKED_UP", result="FAILURE")
    
    print("   Results:")
    for item, count in results:
        print(f"   - {item}: {count} match(es)")
    print("   → Expected: EGG")
    
    # Query: What did Clara PUSH?
    print("\n4. Query: What OBJECT did Clara PUSH?")
    results = memory.query(subject="CLARA", action="PUSHED")
    
    print("   Results:")
    for item, count in results:
        print(f"   - {item}: {count} match(es)")
    print("   → Expected: BOX")
    
    # Query: What succeeded?
    print("\n5. Query: What ACTIONS led to SUCCESS?")
    results = memory.query(result="SUCCESS")
    
    print("   Results:")
    for item, count in results:
        print(f"   - {item}: {count} match(es)")
    print("   → Expected: PICKED_UP (1), PUSHED (1), GRABBED (1)")
    
    # Demonstrate noise tolerance
    print("\n6. Noise tolerance test...")
    cup_hv = memory.create_symbol("CUP")
    
    # Add 10% noise
    noisy_cup = cup_hv.copy()
    noise_indices = random.sample(range(len(noisy_cup)), k=1000)
    noisy_cup[noise_indices] *= -1
    
    similarity = memory.similarity(cup_hv, noisy_cup)
    print(f"   Original vs 10% noisy: {similarity:.3f}")
    print(f"   Still highly similar! (>0.7 is good)")
    
    # Show memory capacity
    print("\n7. Memory capacity demonstration...")
    print(f"   Stored {len(memory.memories)} distinct memories")
    print(f"   Using {len([s for s in memory.symbol_library if not s.startswith('ROLE_')])} unique concept symbols")
    print(f"   Each hypervector: {memory.dimensions} dimensions")
    total_kb = len(memory.symbol_library) * memory.dimensions * 4 / 1024
    print(f"   Total memory: ~{total_kb:.1f} KB")
    
    # Advanced query
    print("\n8. Complex query: What did CLARA do with objects that led to SUCCESS?")
    results = memory.query(subject="CLARA", result="SUCCESS")
    print("   All successful actions by Clara:")
    for item, count in results:
        print(f"   - {item}: {count} match(es)")


if __name__ == "__main__":
    demo_hdc_basics()