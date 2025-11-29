"""
HDC Memory with Neural Embeddings for Natural Language
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoModel, AutoTokenizer

class HDCMemoryWithEmbeddings:
    """HDC Memory that handles natural language via embeddings"""
    
    def __init__(self, hdc_dimensions: int = 10000):
        self.hdc_dimensions = hdc_dimensions
        self.memories = []
        
        # Load a tiny sentence embedding model
        print("Loading embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 22MB, 384 dims
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embedding_dim = 384
        
        # Create random projection matrix to go from 384 → 10,000 dimensions
        # This is created once and reused (so similar embeddings → similar HDC vectors)
        self.projection_matrix = np.random.randn(self.embedding_dim, hdc_dimensions) / np.sqrt(self.embedding_dim)
        
        print(f"Ready! Embedding: {self.embedding_dim} dims → HDC: {hdc_dimensions} dims")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Convert text to neural embedding (384 dims)"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).numpy()[0]
        
        return embedding  # Shape: (384,)
    
    def embedding_to_hdc(self, embedding: np.ndarray) -> np.ndarray:
        """Project neural embedding to HDC hypervector"""
        # Project to high dimension
        hv_continuous = embedding @ self.projection_matrix
        
        # Binarize (threshold at 0)
        hv_binary = np.where(hv_continuous >= 0, 1, -1)
        
        return hv_binary
    
    def encode_text_to_hdc(self, text: str) -> np.ndarray:
        """Full pipeline: text → embedding → HDC hypervector"""
        embedding = self.encode_text(text)
        hdc_vector = self.embedding_to_hdc(embedding)
        return hdc_vector
    
    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine similarity between hypervectors"""
        return np.dot(hv1, hv2) / self.hdc_dimensions
    
    def store_experience(self, text: str, metadata: Dict = None):
        """
        Store a natural language experience
        
        Args:
            text: The experience in natural language
            metadata: Optional structured data (e.g., {'result': 'success', 'emotion': 'happy'})
        """
        # Encode text to HDC
        hv = self.encode_text_to_hdc(text)
        
        self.memories.append({
            'text': text,
            'hv': hv,
            'metadata': metadata or {}
        })
        
        return hv
    
    def query_similar(self, query_text: str, top_k: int = 3, threshold: float = 0.3) -> List[Tuple[str, float, Dict]]:
        """
        Find similar experiences to the query
        
        Returns list of (text, similarity, metadata)
        """
        # Encode query
        query_hv = self.encode_text_to_hdc(query_text)
        
        # Find similar memories
        results = []
        for mem in self.memories:
            sim = self.similarity(query_hv, mem['hv'])
            if sim >= threshold:
                results.append((mem['text'], sim, mem['metadata']))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


def demo_embeddings():
    """Demonstrate HDC with neural embeddings"""
    
    print("=" * 70)
    print("HDC MEMORY WITH NEURAL EMBEDDINGS")
    print("=" * 70)
    
    memory = HDCMemoryWithEmbeddings(hdc_dimensions=10000)
    
    # Store various experiences in natural language
    print("\n1. Storing experiences in natural language...")
    
    experiences = [
        ("Clara picked up the red cup from the table", {'result': 'success'}),
        ("Clara grabbed the mug and lifted it successfully", {'result': 'success'}),
        ("Clara tried to pick up the egg but it broke", {'result': 'failure'}),
        ("The robot pushed the cardboard box across the room", {'result': 'success'}),
        ("I helped the user debug their Python async code", {'result': 'success', 'topic': 'programming'}),
        ("The user asked about asynchronous programming in Python", {'topic': 'programming'}),
        ("We discussed how event loops work in asyncio", {'topic': 'programming'}),
        ("Clara detected an obstacle and stopped moving", {'result': 'success', 'topic': 'navigation'}),
    ]
    
    for text, metadata in experiences:
        memory.store_experience(text, metadata)
        print(f"   ✓ Stored: {text[:50]}...")
    
    # Query 1: Similar phrasing
    print("\n2. Query: 'Clara lifted a cup' (similar to stored experience)")
    results = memory.query_similar("Clara lifted a cup", top_k=3)
    
    print("   Top matches:")
    for text, sim, meta in results:
        print(f"   - [{sim:.3f}] {text[:60]}...")
    
    # Query 2: Semantic similarity (different words, same meaning)
    print("\n3. Query: 'Help me understand Python async' (semantic match)")
    results = memory.query_similar("Help me understand Python async", top_k=3)
    
    print("   Top matches:")
    for text, sim, meta in results:
        print(f"   - [{sim:.3f}] {text[:60]}...")
    
    # Query 3: Different domain
    print("\n4. Query: 'The robot avoided hitting something' (navigation)")
    results = memory.query_similar("The robot avoided hitting something", top_k=3)
    
    print("   Top matches:")
    for text, sim, meta in results:
        print(f"   - [{sim:.3f}] {text[:60]}...")
    
    # Query 4: Failures specifically
    print("\n5. Query: 'What went wrong with the egg?'")
    results = memory.query_similar("What went wrong with the egg?", top_k=3)
    
    print("   Top matches:")
    for text, sim, meta in results:
        result = meta.get('result', 'unknown')
        print(f"   - [{sim:.3f}] {text[:60]}... (result: {result})")
    
    # Show the power: completely different phrasing
    print("\n6. Demonstrating semantic understanding:")
    print("   Storing: 'The user was frustrated with their code'")
    memory.store_experience("The user was frustrated with their code", {'emotion': 'frustrated'})
    
    print("   Query: 'Someone is having trouble programming'")
    results = memory.query_similar("Someone is having trouble programming", top_k=1)
    
    for text, sim, meta in results:
        print(f"   Match: [{sim:.3f}] {text}")
        print(f"   → HDC found the match even though no words overlap!")


if __name__ == "__main__":
    demo_embeddings()