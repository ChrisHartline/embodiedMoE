"""
Clara v2: With Neural Embeddings for Natural Language Memory
"""

import numpy as np
from typing import Dict, List, Optional
import torch
from transformers import AutoModel, AutoTokenizer

class ClaraWithEmbeddings:
    """Clara with HDC + Neural Embeddings"""
    
    def __init__(self):
        print("Initializing Clara v2.0...")
        
        # Load embedding model
        print("  Loading neural embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embedding_dim = 384
        
        # HDC settings
        self.hdc_dimensions = 10000
        self.projection_matrix = np.random.randn(self.embedding_dim, self.hdc_dimensions) / np.sqrt(self.embedding_dim)
        
        # Memory systems
        self.episodic_memory = []  # Recent experiences
        self.user_profile = []     # Things learned about user
        
        # Personality (Clara's defaults)
        self.personality = {
            'warmth': 0.8,
            'formality': 0.3,
            'verbosity': 0.6,
            'patience': 0.9
        }
        
        # Interaction counter
        self.interaction_count = 0
        
        print("  Clara v2.0 ready!")
        print(f"  Personality: warmth={self.personality['warmth']}, "
              f"formality={self.personality['formality']}")
    
    def encode_text_to_hdc(self, text: str) -> np.ndarray:
        """Convert text to HDC hypervector via neural embedding"""
        # Get neural embedding
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
        
        # Project to HDC
        hv_continuous = embedding @ self.projection_matrix
        hv_binary = np.where(hv_continuous >= 0, 1, -1)
        
        return hv_binary
    
    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine similarity"""
        return np.dot(hv1, hv2) / self.hdc_dimensions
    
    def store_experience(self, text: str, category: str = 'general', metadata: Dict = None):
        """Store an experience in episodic memory"""
        hv = self.encode_text_to_hdc(text)
        
        self.episodic_memory.append({
            'text': text,
            'hv': hv,
            'category': category,
            'metadata': metadata or {},
            'interaction_num': self.interaction_count
        })
        
        # Keep only last 50 experiences (to simulate working memory)
        if len(self.episodic_memory) > 50:
            self.episodic_memory.pop(0)
    
    def retrieve_similar_experiences(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """Find similar past experiences"""
        query_hv = self.encode_text_to_hdc(query)
        
        results = []
        for mem in self.episodic_memory:
            sim = self.similarity(query_hv, mem['hv'])
            if sim >= threshold:
                results.append({
                    'text': mem['text'],
                    'similarity': sim,
                    'category': mem['category'],
                    'metadata': mem['metadata']
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def learn_about_user(self, observation: str):
        """Store something learned about the user"""
        hv = self.encode_text_to_hdc(observation)
        
        self.user_profile.append({
            'text': observation,
            'hv': hv
        })
        
        print(f"  [Clara learned: {observation}]")
    
    def get_user_context(self, query: str) -> List[str]:
        """Retrieve relevant user preferences/context"""
        query_hv = self.encode_text_to_hdc(query)
        
        relevant = []
        for profile_item in self.user_profile:
            sim = self.similarity(query_hv, profile_item['hv'])
            if sim >= 0.3:
                relevant.append(profile_item['text'])
        
        return relevant
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate Clara's response
        (Simplified - in full version would use tiny LM)
        """
        self.interaction_count += 1
        
        # Retrieve similar past experiences
        similar = self.retrieve_similar_experiences(user_input, top_k=2)
        
        # Get user context
        user_context = self.get_user_context(user_input)
        
        # Simple rule-based response (would be tiny LM in full version)
        response = self._generate_base_response(user_input, similar, user_context)
        
        # Apply personality
        response = self._apply_personality(response)
        
        # Store this interaction
        self.store_experience(
            f"User said: {user_input}. Clara responded: {response}",
            category='conversation'
        )
        
        return response
    
    def _generate_base_response(self, user_input: str, similar_experiences: List, user_context: List) -> str:
        """Generate base response (simplified)"""
        
        input_lower = user_input.lower()
        
        # Check if we have similar past experiences
        if similar_experiences and similar_experiences[0]['similarity'] > 0.5:
            past = similar_experiences[0]
            return f"This reminds me of when we talked about this before. {self._contextual_response(input_lower)}"
        
        # Check user context
        if user_context:
            return f"Based on what I know about you, {self._contextual_response(input_lower)}"
        
        # Default responses
        return self._contextual_response(input_lower)
    
    def _contextual_response(self, input_lower: str) -> str:
        """Generate contextual response based on input"""
        
        if 'async' in input_lower or 'asyncio' in input_lower:
            return "I can help you understand async programming. It's like having multiple tasks running concurrently."
        
        elif 'help' in input_lower or 'stuck' in input_lower:
            return "I'd be happy to help! What specifically are you working on?"
        
        elif 'thanks' in input_lower or 'thank you' in input_lower:
            return "You're very welcome! I enjoy helping you."
        
        elif 'robot' in input_lower or 'pick up' in input_lower or 'grasp' in input_lower:
            return "I can help with robotic manipulation. What object are you working with?"
        
        else:
            return "That's interesting! Tell me more about what you're thinking."
    
    def _apply_personality(self, response: str) -> str:
        """Apply Clara's personality to response"""
        
        # Apply warmth
        if self.personality['warmth'] > 0.7:
            if not response.endswith('!'):
                # Add enthusiasm
                response = response.rstrip('.') + '!'
        
        # Apply formality (low = casual)
        if self.personality['formality'] < 0.4:
            response = response.replace('I would', "I'd")
            response = response.replace('I will', "I'll")
        
        return response
    
    def respond(self, user_input: str) -> str:
        """Main interface: user talks to Clara"""
        
        print(f"\nUser: {user_input}")
        
        response = self.generate_response(user_input)
        
        print(f"Clara: {response}")
        
        return response
    
    def adapt_to_feedback(self, feedback: str):
        """Learn from user feedback"""
        
        feedback_lower = feedback.lower()
        
        if 'concise' in feedback_lower or 'shorter' in feedback_lower:
            self.personality['verbosity'] = max(0.1, self.personality['verbosity'] - 0.2)
            self.learn_about_user("User prefers concise responses")
            print("  [Adjusting: being more concise]")
        
        elif 'detail' in feedback_lower or 'more' in feedback_lower:
            self.personality['verbosity'] = min(0.9, self.personality['verbosity'] + 0.2)
            self.learn_about_user("User prefers detailed explanations")
            print("  [Adjusting: providing more detail]")
    
    def show_memory_stats(self):
        """Display memory statistics"""
        
        print("\n" + "=" * 60)
        print("CLARA'S MEMORY")
        print("=" * 60)
        print(f"Episodic memories: {len(self.episodic_memory)}")
        print(f"User profile items: {len(self.user_profile)}")
        print(f"Total interactions: {self.interaction_count}")
        
        if self.user_profile:
            print("\nWhat Clara knows about you:")
            for item in self.user_profile:
                print(f"  - {item['text']}")


def demo_clara_v2():
    """Demonstrate Clara with neural embeddings"""
    
    print("=" * 60)
    print("CLARA v2.0 - WITH NEURAL EMBEDDINGS")
    print("=" * 60)
    
    clara = ClaraWithEmbeddings()
    
    # Conversation 1: First interaction
    print("\n--- Day 1: First Conversation ---")
    clara.respond("Hi Clara, I'm struggling with async programming in Python")
    clara.respond("Can you explain how event loops work?")
    
    # Clara learns something about the user
    clara.learn_about_user("User is learning Python async programming")
    clara.learn_about_user("User prefers practical examples")
    
    # Conversation 2: Later topic
    print("\n--- Day 1: Different Topic ---")
    clara.respond("Can you help me with robot grasping?")
    clara.respond("The robot keeps dropping fragile objects")
    
    # Conversation 3: Back to Python (should remember!)
    print("\n--- Day 2: Back to Python ---")
    clara.respond("I'm still confused about asyncio")
    
    # User gives feedback
    print("\n--- User Feedback ---")
    print("User: Can you be more concise?")
    clara.adapt_to_feedback("Can you be more concise?")
    
    # See the change
    print("\n--- After Feedback ---")
    clara.respond("Explain async to me again")
    
    # Show what Clara remembers
    clara.show_memory_stats()
    
    # Demonstrate memory retrieval
    print("\n" + "=" * 60)
    print("MEMORY RETRIEVAL TEST")
    print("=" * 60)
    
    print("\nQuery: 'What did we discuss about Python?'")
    results = clara.retrieve_similar_experiences("What did we discuss about Python?", top_k=3)
    print("Clara's memories:")
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['text'][:70]}...")
    
    print("\nQuery: 'What problems did the robot have?'")
    results = clara.retrieve_similar_experiences("What problems did the robot have?", top_k=3)
    print("Clara's memories:")
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['text'][:70]}...")


if __name__ == "__main__":
    demo_clara_v2()