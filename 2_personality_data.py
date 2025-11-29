"""
Generate training data for personality components
Uses Claude API to create transformation pairs
"""

import json
import os
from typing import List, Dict
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def generate_personality_examples(
    dimension: str,
    n_examples: int = 50,
    api_key: str = None
) -> List[Dict]:
    """
    Generate training examples for a personality dimension
    
    Args:
        dimension: 'warmth', 'formality', or 'verbosity'
        n_examples: Number of examples to generate
        api_key: Anthropic API key (or uses .env)
    """
    
    # Use .env key if not provided
    if api_key is None:
        api_key = os.getenv('CLAUDE_API_KEY')
    
    if not api_key:
        raise ValueError(
            "No API key found! Set CLAUDE_API_KEY in .env file or pass api_key parameter"
        )
    
    client = Anthropic(api_key=api_key)
    
    # Base neutral statements
    neutral_statements = [
        "That is correct.",
        "I can help with that.",
        "Here's how to fix the error.",
        "This approach won't work.",
        "The answer is 42.",
        "Your code has a bug.",
        "Let me explain this concept.",
        "That's an interesting question.",
        "I don't understand what you mean.",
        "This will take some time.",
        "The function returns a value.",
        "You need to install the library.",
        "The loop runs five times.",
        "There's a syntax error.",
        "The variable is undefined.",
    ] * (n_examples // 15 + 1)
    
    neutral_statements = neutral_statements[:n_examples]
    
    # Prompts for each dimension
    prompts = {
        'warmth': {
            'low': "Rewrite this to be cold, distant, and minimally friendly (keep it brief): '{text}'\n\nJust provide the rewritten text, nothing else.",
            'high': "Rewrite this to be warm, friendly, and enthusiastic (keep meaning the same): '{text}'\n\nJust provide the rewritten text, nothing else."
        },
        'formality': {
            'low': "Rewrite this to be very casual and informal: '{text}'\n\nJust provide the rewritten text, nothing else.",
            'high': "Rewrite this to be formal and professional: '{text}'\n\nJust provide the rewritten text, nothing else."
        },
        'verbosity': {
            'low': "Rewrite this to be extremely concise (under 8 words): '{text}'\n\nJust provide the rewritten text, nothing else.",
            'high': "Rewrite this to be detailed and explanatory (15-25 words): '{text}'\n\nJust provide the rewritten text, nothing else."
        }
    }
    
    if dimension not in prompts:
        raise ValueError(f"Unknown dimension: {dimension}. Choose from: {list(prompts.keys())}")
    
    examples = []
    
    print(f"Generating {n_examples} examples for '{dimension}' dimension...")
    print(f"This will make {n_examples * 2} API calls and may take a few minutes.")
    print()
    
    for i, neutral in enumerate(neutral_statements):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_examples}")
        
        try:
            # Generate low version
            low_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": prompts[dimension]['low'].format(text=neutral)
                }]
            )
            low_text = low_response.content[0].text.strip()
            
            # Generate high version
            high_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": prompts[dimension]['high'].format(text=neutral)
                }]
            )
            high_text = high_response.content[0].text.strip()
            
            examples.append({
                'neutral': neutral,
                'low': low_text,
                'high': high_text,
                'dimension': dimension
            })
            
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            continue
    
    print(f"\nGenerated {len(examples)} examples successfully!")
    return examples


def save_examples(examples: List[Dict], filename: str):
    """Save examples to JSON file"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    filepath = os.path.join('data', filename)
    with open(filepath, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Saved to {filepath}")


def demo_generate_data():
    """Demo: Generate a small dataset"""
    
    print("=" * 60)
    print("PERSONALITY DATA GENERATION")
    print("=" * 60)
    
    # Check for API key
    load_dotenv()
    api_key = os.getenv('CLAUDE_API_KEY')
    
    if not api_key:
        print("\n⚠️  WARNING: No CLAUDE_API_KEY found in .env file")
        print("\nCreate a .env file with:")
        print("CLAUDE_API_KEY=your_key_here")
        print("\nUsing mock data instead for demonstration...\n")
        
        # Create mock data for demo
        mock_examples = [
            {
                'neutral': 'That is correct.',
                'low': 'Correct.',
                'high': "That's exactly right! Great job!",
                'dimension': 'warmth'
            },
            {
                'neutral': 'I can help with that.',
                'low': 'Sure.',
                'high': "I'd be absolutely delighted to help you with that!",
                'dimension': 'warmth'
            },
            {
                'neutral': "Here's how to fix the error.",
                'low': 'Fix it like this.',
                'high': "I'd be happy to show you how to fix that error!",
                'dimension': 'warmth'
            }
        ]
        save_examples(mock_examples, 'mock_warmth_data.json')
        return
    
    # Generate real data (small batch for demo)
    print("\nGenerating warmth training data (10 examples)...")
    print("Cost estimate: ~$0.05")
    
    response = input("\nProceed with API calls? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    warmth_data = generate_personality_examples(
        dimension='warmth',
        n_examples=10,  # Start small for demo
        api_key=api_key
    )
    
    save_examples(warmth_data, 'warmth_training_data.json')
    
    print("\n" + "=" * 60)
    print("EXAMPLE TRANSFORMATIONS:")
    print("=" * 60)
    for i, ex in enumerate(warmth_data[:3], 1):
        print(f"\n{i}. Neutral:     {ex['neutral']}")
        print(f"   Low warmth:  {ex['low']}")
        print(f"   High warmth: {ex['high']}")


if __name__ == "__main__":
    demo_generate_data()