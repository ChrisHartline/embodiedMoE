"""
Clara Prototype - Combining HDC Memory + Tiny LM
This is a simplified but functional version
"""

import numpy as np
from typing import Dict, List, Optional
import random

# Import our HDC system from file 1
import sys
sys.path.append('.')
from hdc_basics import HDCMemory

class PersonalityTransformer:
    """
    Simulated personality transformation
    In reality, this would be a trained neural module
    """
    
    def __init__(self, dimension: str):
        self.dimension = dimension
    
    def apply(self, text: str, level: float) -> str:
        """
        Transform text along personality dimension
        This is a rule-based simulation - in practice would be learned
        """
        
        if self.dimension == 'warmth':
            if level > 0.7:
                # Add warm markers
                if text.startswith("Here's"):
                    text = "I'd be happy to help! " + text
                if "error" in text.lower():
                    text = text.replace("error", "small issue")
                text = text.rstrip('.') + "!"
            elif level < 0.3:
                # Make colder
                text = text.replace("!", ".")
                text = text.replace(" I'd", " I will")
        
        elif self.dimension == 'formality':
            if level > 0.7:
                # More formal
                text = text.replace("gonna", "going to")
                text = text.replace("can't", "cannot")
            elif level < 0.3:
                # More casual
                text = text.replace("cannot", "can't")
                if not text.endswith(('!', '?')):
                    text = text.rstrip('.') + '.'
        
        elif self.dimension == 'verbosity':
            if level < 0.3:
                # More concise - take first sentence only
                sentences = text.split('.')
                text = sentences[0] + '.'
            elif level > 0.7:
                # More verbose - add explanation
                text = text + " Let me explain in more detail."
        
        return text


class ClaraPrototype:
    """
    Clara: Embodied AI with HDC memory and personality
    """
    
    def __init__(self):
        # HDC memory system
        self.hdc_memory = HDCMemory(dimensions=10000)
        
        # User-specific memory
        self.user_memory = HDCMemory(dimensions=10000)
        
        # Personality settings (Clara's defaults)
        self.personality = {
            'warmth': 0.8,
            'formality': 0.3,
            'verbosity': 0.6
        }
        
        # Personality transformers
        self.transformers = {
            'warmth': PersonalityTransformer('warmth'),
            'formality': PersonalityTransformer('formality'),
            'verbosity': PersonalityTransformer('verbosity')
        }
        
        # Interaction history
        self.interactions = []
        
        print("Clara initialized!")
        print(f"Personality: warmth={self.personality['warmth']}, "
              f"formality={self.personality['formality']}, "
              f"verbosity={self.personality['verbosity']}")
    
    def store_knowledge(self, *concepts: str):
        """Store a fact in knowledge base"""
        self.hdc_memory.store_memory(*concepts)
        print(f"  [Stored: {' ⊗ '.join(concepts)}]")
    
    def store_user_preference(self, *concepts: str):
        """Store user-specific preference"""
        self.user_memory.store_memory(*concepts)
        print(f"  [Learned: {' ⊗ '.join(concepts)}]")
    
    def retrieve_knowledge(self, *query_concepts: str) -> List[tuple]:
        """Query knowledge base"""
        result = self.hdc_memory.query(*query_concepts)
        if result is not None:
            return self.hdc_memory.find_closest_symbol(result, top_k=3)
        return []
    
    def retrieve_user_context(self, query: str) -> Dict:
        """Retrieve user preferences relevant to query"""
        # Simplified: just return current settings
        return {
            'expertise_level': 'intermediate',
            'prefers_examples': True
        }
    
    def generate_base_response(self, query: str) -> str:
        """
        Generate base response (simulated tiny LM)
        In practice, this would call the actual tiny language model
        """
        
        # Simple rule-based responses for demo
        query_lower = query.lower()
        
        if 'async' in query_lower or 'asyncio' in query_lower:
            knowledge = self.retrieve_knowledge("ASYNC", "PYTHON")
            return "Here's how async works in Python. It uses an event loop to manage concurrent tasks."
        
        elif 'bug' in query_lower or 'error' in query_lower:
            return "Here's how to fix that error. Check your code on line 5."
        
        elif 'help' in query_lower:
            return "I can help with that. What specifically are you working on?"
        
        elif 'learn' in query_lower or 'teach' in query_lower:
            return "I can teach you that concept. It works like this."
        
        else:
            return "That's an interesting question. Let me think about it."
    
    def apply_personality(self, base_response: str, context: Dict) -> str:
        """Apply personality transformations"""
        
        response = base_response
        
        # Apply each personality dimension
        for dimension, transformer in self.transformers.items():
            level = self.personality[dimension]
            
            # Adjust based on user context
            if dimension == 'verbosity' and context.get('expertise_level') == 'expert':
                level *= 0.5  # Be more concise for experts
            
            response = transformer.apply(response, level)
        
        return response
    
    def respond(self, query: str) -> str:
        """Generate Clara's response to user query"""
        
        print(f"\nUser: {query}")
        
        # 1. Retrieve user context
        context = self.retrieve_user_context(query)
        
        # 2. Generate base response
        base = self.generate_base_response(query)
        print(f"  [Base: {base}]")
        
        # 3. Apply personality
        response = self.apply_personality(base, context)
        
        # 4. Store interaction
        self.interactions.append({
            'query': query,
            'response': response
        })
        
        print(f"Clara: {response}")
        return response
    
    def learn_from_feedback(self, feedback: str):
        """Learn from user feedback"""
        
        if 'concise' in feedback.lower() or 'shorter' in feedback.lower():
            print("  [Adjusting: lowering verbosity]")
            self.personality['verbosity'] = max(0.1, self.personality['verbosity'] - 0.2)
            self.store_user_preference("USER", "PREFERS", "CONCISE")
        
        elif 'more detail' in feedback.lower() or 'explain more' in feedback.lower():
            print("  [Adjusting: increasing verbosity]")
            self.personality['verbosity'] = min(0.9, self.personality['verbosity'] + 0.2)
            self.store_user_preference("USER", "PREFERS", "DETAILED")
        
        elif 'warm' in feedback.lower() or 'friendly' in feedback.lower():
            print("  [Adjusting: increasing warmth]")
            self.personality['warmth'] = min(0.9, self.personality['warmth'] + 0.1)
    
    def sleep(self):
        """Simulate nightly consolidation"""
        print("\n" + "=" * 60)
        print("Clara: Going to sleep...")
        print("=" * 60)
        
        print(f"\nProcessed {len(self.interactions)} interactions today")
        
        # Simulate pattern extraction
        print("\nConsolidating memories...")
        if len(self.interactions) > 0:
            print(f"  - Found {len(self.interactions)} experiences")
            print(f"  - Current personality: {self.personality}")
        
        print("\nClara: Sleep complete. Ready for tomorrow!")
        print("=" * 60)


def demo_clara():
    """Demonstrate Clara in action"""
    
    print("=" * 60)
    print("CLARA PROTOTYPE DEMO")
    print("=" * 60)
    
    # Initialize Clara
    clara = ClaraPrototype()
    
    # Teach Clara some facts
    print("\n1. Teaching Clara...")
    clara.store_knowledge("TOKIO", "IS_A", "ASYNC_RUNTIME", "FOR", "RUST")
    clara.store_knowledge("PYTHON", "USES", "ASYNCIO", "FOR", "ASYNC")
    
    # Conversation
    print("\n2. Conversation...")
    clara.respond("Can you help me with async in Python?")
    clara.respond("I have a bug in my code")
    clara.respond("How does asyncio work?")
    
    # User feedback
    print("\n3. User provides feedback...")
    print("User: Can you be more concise?")
    clara.learn_from_feedback("Can you be more concise?")
    
    # See personality change
    print("\n4. Response after feedback...")
    clara.respond("Can you help me with async in Python?")
    
    # Sleep/consolidation
    print("\n5. End of day...")
    clara.sleep()


if __name__ == "__main__":
    demo_clara()