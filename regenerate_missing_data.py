# regenerate_missing_data.py
import json
import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
import time

load_dotenv()

client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

def generate_domain_data(domain: str, n_examples: int = 1000) -> list:
    """Generate domain knowledge examples"""
    
    domain_focus = {
        'coding': 'Python programming, async/await, debugging, best practices, code explanations, data structures, algorithms',
        'teaching': 'explaining concepts clearly, using analogies, scaffolding learning, patience, checking understanding, pedagogy',
        'quantum': 'Qiskit, quantum circuits, qubits, superposition, entanglement, linear algebra for QC, quantum gates, quantum mechanics'
    }
    
    focus = domain_focus.get(domain, f'{domain} concepts')
    
    examples = []
    batch_size = 10
    
    for batch_start in range(0, n_examples, batch_size):
        batch_num = batch_start // batch_size + 1
        total_batches = (n_examples + batch_size - 1) // batch_size
        
        print(f"  Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        
        prompt = f"""Generate {batch_size} Q&A training examples for the '{domain}' knowledge domain.

Focus areas: {focus}

For each example, provide:
1. QUESTION: A question someone might ask about {domain}
2. ANSWER: A clear, helpful answer (2-4 sentences)
3. DIFFICULTY: beginner, intermediate, or advanced

Mix of difficulty levels. Make questions realistic and answers educational.

Return ONLY a JSON array, no other text:
[
  {{"question": "...", "answer": "...", "difficulty": "beginner"}},
  {{"question": "...", "answer": "...", "difficulty": "intermediate"}}
]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            text = text.replace('```json', '').replace('```', '').strip()
            
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                text = text[start_idx:end_idx]
            
            batch_examples = json.loads(text)
            
            for ex in batch_examples:
                ex['domain'] = domain
            
            examples.extend(batch_examples)
            print(f"✓ ({len(examples)} total)")
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue
        
        time.sleep(1)
    
    return examples[:n_examples]


# Generate missing data
data_dir = Path('./data')

missing_domains = ['coding', 'teaching', 'quantum']

for domain in missing_domains:
    output_file = data_dir / f"{domain}_knowledge.json"
    
    # Check if already has data
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        if len(existing) > 100:
            print(f"✓ {domain}: Already has {len(existing)} examples, skipping")
            continue
    
    print(f"\n🔄 Generating {domain} data...")
    examples = generate_domain_data(domain, n_examples=1000)
    
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✓ Saved {len(examples)} examples to {output_file}")

print("\n✓ Done! Upload the updated files to Google Drive.")