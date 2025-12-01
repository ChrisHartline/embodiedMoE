"""
Clara Master Pipeline - Complete Automation
Creates Clara from specification to deployment
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()


class ClaraMasterPipeline:
    """
    Complete automated pipeline for Clara creation
    """
    
    def __init__(self, spec_file: str = "clara_spec.json"):
        """
        Initialize with a specification file
        """
        # Load specification
        if not Path(spec_file).exists():
            print(f"Creating default spec: {spec_file}")
            self.create_default_spec(spec_file)
        
        with open(spec_file) as f:
            self.spec = json.load(f)
        
        # Setup directories
        self.dirs = {
            'data': Path('./data'),
            'models': Path('./models'),
            'configs': Path('./configs'),
            'results': Path('./results')
        }
        
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)
        
        # Initialize monitoring (with error handling)
        self.monitor = None
        try:
            from clara_monitoring_complete import ClaraMonitor
            self.monitor = ClaraMonitor(project_name="clara-deng-research", entity="chris_hartline")
        except Exception as e:
            print(f"⚠️  Monitoring not available: {e}")
            print("   Continuing without W&B tracking")
        
        self._print_header()
    
    def _print_header(self):
        """Print initialization header"""
        print("=" * 70)
        print("CLARA MASTER PIPELINE INITIALIZED")
        print("=" * 70)
        print(f"\nPersonality dimensions: {list(self.spec['personality'].keys())}")
        print(f"Domain expertise: {list(self.spec['expertise'].keys())}")
        print(f"Base model: {self.spec['config']['base_model']}")
        print(f"Examples per dimension: {self.spec['config']['examples_per_dimension']}")
    
    def create_default_spec(self, spec_file: str):
        """Create default Clara specification"""
        spec = {
            "personality": {
                "warmth": 0.8,
                "playful": 0.7,
                "formal": 0.3,
                "encouragement": 0.9
            },
            "expertise": {
                "medical": 0.7,
                "coding": 0.8,
                "teaching": 0.9,
                "quantum": 0.8
            },
            "config": {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "examples_per_dimension": 1000,
                "merge_method": "ties"
            }
        }
        
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"✓ Created {spec_file}")
    
    def run_complete_pipeline(self, skip_training: bool = False):
        """
        Execute the complete pipeline end-to-end
        """
        print("\n" + "=" * 70)
        print("STARTING COMPLETE PIPELINE")
        print("=" * 70)
        
        # Phase 1: Already done (spec loaded)
        print("\n✓ Phase 1: Specification loaded")
        
        # Phase 2: Data Generation
        print("\n" + "=" * 70)
        print("PHASE 2: DATA GENERATION")
        print("=" * 70)
        
        data_files = self.generate_all_training_data()
        
        print("\n" + "=" * 70)
        print("DATA GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated/found {len(data_files)} data files:")
        for name, path in data_files.items():
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"  ✓ {name}: {path} ({size_kb:.1f} KB)")
            else:
                print(f"  ❌ {name}: {path} (NOT FOUND)")
        
        # Phase 3: Fine-tuning Instructions
        if not skip_training:
            print("\n" + "=" * 70)
            print("PHASE 3: FINE-TUNING (COLAB)")
            print("=" * 70)
            
            self.prepare_finetuning_instructions()
            
            print("\n⚠️  MANUAL STEP REQUIRED:")
            print("   1. Upload data files to Google Drive")
            print("   2. Open Colab notebooks in ./configs/")
            print("   3. Run training for each dimension")
            print("   4. Download models to ./models/")
            print("   5. Re-run: python clara_master.py")
            print("\n   (Models will be detected and training skipped)")
            return
        
        # Continue with merging if models exist
        print("\n✓ Models found, continuing to merge phase...")
        # ... rest of pipeline
    
    def generate_all_training_data(self) -> Dict[str, Path]:
        """
        Phase 2: Generate training data for all dimensions
        Skips existing data files
        """
        from anthropic import Anthropic
        
        n_examples = self.spec['config']['examples_per_dimension']
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            print("❌ ERROR: No ANTHROPIC_API_KEY found!")
            print("   Set it in .env file: ANTHROPIC_API_KEY=your_key_here")
            return {}
        
        client = Anthropic(api_key=api_key)
        data_files = {}
        
        # Generate personality data
        print("\n--- PERSONALITY DATA ---")
        for dimension in self.spec['personality'].keys():
            data_file = self.dirs['data'] / f"{dimension}_training.json"
            
            # Check if data already exists
            if data_file.exists():
                with open(data_file) as f:
                    existing = json.load(f)
                
                if len(existing) >= n_examples * 0.9:  # At least 90% complete
                    print(f"\n✓ SKIP {dimension}: Found {len(existing)} existing examples")
                    data_files[dimension] = data_file
                    continue
                else:
                    print(f"\n⚠️  {dimension}: Only {len(existing)} examples, regenerating...")
            
            print(f"\n🔄 GENERATING {dimension} ({n_examples} examples)...")
            
            examples = self.generate_personality_data(client, dimension, n_examples)
            
            # Save
            with open(data_file, 'w') as f:
                json.dump(examples, f, indent=2)
            
            print(f"   ✓ Saved {len(examples)} examples to {data_file}")
            data_files[dimension] = data_file
            
            # Track in W&B
            self._track_data(dimension, examples)
        
        # Generate domain data
        print("\n--- DOMAIN KNOWLEDGE DATA ---")
        for domain in self.spec['expertise'].keys():
            data_file = self.dirs['data'] / f"{domain}_knowledge.json"
            
            # Check if data already exists
            if data_file.exists():
                with open(data_file) as f:
                    existing = json.load(f)
                
                if len(existing) >= n_examples * 0.9:
                    print(f"\n✓ SKIP {domain}: Found {len(existing)} existing examples")
                    data_files[domain] = data_file
                    continue
                else:
                    print(f"\n⚠️  {domain}: Only {len(existing)} examples, regenerating...")
            
            print(f"\n🔄 GENERATING {domain} ({n_examples} examples)...")
            
            examples = self.generate_domain_data(client, domain, n_examples)
            
            # Save
            with open(data_file, 'w') as f:
                json.dump(examples, f, indent=2)
            
            print(f"   ✓ Saved {len(examples)} examples to {data_file}")
            data_files[domain] = data_file
            
            # Track in W&B
            self._track_data(domain, examples)
        
        return data_files
    
    def _track_data(self, dimension: str, examples: List[Dict]):
        """Track data generation in W&B (with error handling)"""
        if self.monitor is None:
            return
        
        try:
            if hasattr(self.monitor, 'track_data_generation'):
                n = len(examples)
                cost = (n * 2 * 0.003 / 1000)  # Rough estimate
                self.monitor.track_data_generation(
                    dimension=dimension,
                    n_examples=n,
                    api_cost=cost,
                    examples=examples
                )
        except Exception as e:
            print(f"   ⚠️  W&B tracking error: {e}")
    
    def generate_personality_data(
        self,
        client,
        dimension: str,
        n_examples: int
    ) -> List[Dict]:
        """Generate personality training examples via Claude API"""
        
        examples = []
        batch_size = 10
        
        # Dimension-specific prompts
        dimension_descriptions = {
            'warmth': 'warm, friendly, caring vs cold, distant, detached',
            'playful': 'fun, witty, lighthearted vs serious, straightforward, no-nonsense',
            'formal': 'professional, proper, structured vs casual, relaxed, conversational',
            'encouragement': 'supportive, motivating, uplifting vs neutral, matter-of-fact, reserved'
        }
        
        desc = dimension_descriptions.get(dimension, f'high {dimension} vs low {dimension}')
        
        for batch_start in range(0, n_examples, batch_size):
            batch_num = batch_start // batch_size + 1
            total_batches = (n_examples + batch_size - 1) // batch_size
            
            print(f"   Batch {batch_num}/{total_batches}...", end=" ", flush=True)
            
            prompt = f"""Generate {batch_size} training examples for the '{dimension}' personality dimension.

This dimension represents: {desc}

For each example, provide THREE versions of the same message:
1. NEUTRAL: Balanced, professional baseline
2. LOW: Minimal {dimension} (e.g., {desc.split(' vs ')[1]})
3. HIGH: Maximum {dimension} (e.g., {desc.split(' vs ')[0]})

Examples should be varied: greetings, responses, explanations, questions, encouragements.
Each should be 1-3 sentences.

Return ONLY a JSON array, no other text:
[
  {{"neutral": "...", "low": "...", "high": "..."}},
  {{"neutral": "...", "low": "...", "high": "..."}}
]"""

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse JSON response
                text = response.content[0].text
                text = text.replace('```json', '').replace('```', '').strip()
                
                # Find the JSON array
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    text = text[start_idx:end_idx]
                
                batch_examples = json.loads(text)
                
                # Add dimension tag
                for ex in batch_examples:
                    ex['dimension'] = dimension
                
                examples.extend(batch_examples)
                print(f"✓ ({len(examples)} total)")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error, retrying...")
                continue
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
        
        return examples[:n_examples]
    
    def generate_domain_data(
        self,
        client,
        domain: str,
        n_examples: int
    ) -> List[Dict]:
        """Generate domain knowledge examples via Claude API"""
        
        examples = []
        batch_size = 10
        
        # Domain-specific focus areas
        domain_focus = {
            'medical': 'medical terminology, anatomy, patient care, clinical reasoning, symptoms and treatments',
            'coding': 'Python programming, async/await, debugging, best practices, code explanations',
            'teaching': 'explaining concepts clearly, using analogies, scaffolding learning, patience, checking understanding',
            'quantum': 'Qiskit, quantum circuits, qubits, superposition, entanglement, linear algebra for QC, quantum gates'
        }
        
        focus = domain_focus.get(domain, f'{domain} concepts and applications')
        
        for batch_start in range(0, n_examples, batch_size):
            batch_num = batch_start // batch_size + 1
            total_batches = (n_examples + batch_size - 1) // batch_size
            
            print(f"   Batch {batch_num}/{total_batches}...", end=" ", flush=True)
            
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
                
                # Parse JSON response
                text = response.content[0].text
                text = text.replace('```json', '').replace('```', '').strip()
                
                # Find the JSON array
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    text = text[start_idx:end_idx]
                
                batch_examples = json.loads(text)
                
                # Add domain tag
                for ex in batch_examples:
                    ex['domain'] = domain
                
                examples.extend(batch_examples)
                print(f"✓ ({len(examples)} total)")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error, retrying...")
                continue
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
        
        return examples[:n_examples]
    
    def prepare_finetuning_instructions(self):
        """Generate instructions and notebooks for Colab training"""
        
        print("\nGenerating Colab training instructions...")
        
        instructions = f"""
# Clara Fine-tuning Instructions

## Data Files (upload to Google Drive)
"""
        
        for dim in self.spec['personality'].keys():
            instructions += f"- ./data/{dim}_training.json\n"
        
        for domain in self.spec['expertise'].keys():
            instructions += f"- ./data/{domain}_knowledge.json\n"
        
        instructions += f"""
## Training Steps

1. Upload all data files to: Google Drive/clara_data/

2. For each dimension/domain, run training in Colab:
   - Base model: {self.spec['config']['base_model']}
   - Method: LoRA (r=16, alpha=32)
   - Epochs: 3
   - Batch size: 4 (effective 16 with grad accumulation)

3. Save models to: Google Drive/clara_models/

4. Download to local: ./models/tinyllama_<dimension>/

## Colab Notebook Template

See: ./configs/clara_training_template.ipynb
"""
        
        # Save instructions
        instructions_file = self.dirs['results'] / "TRAINING_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"✓ Saved: {instructions_file}")


def main():
    """Main entry point"""
    
    # Check for spec file
    spec_file = "clara_spec.json"
    
    # Create or update spec with your preferences
    spec = {
        "personality": {
            "warmth": 0.8,
            "playful": 0.7,
            "formal": 0.3,
            "encouragement": 0.9
        },
        "expertise": {
            "medical": 0.7,
            "coding": 0.8,
            "teaching": 0.9,
            "quantum": 0.8
        },
        "config": {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "examples_per_dimension": 1000,
            "merge_method": "ties"
        }
    }
    
    with open(spec_file, 'w') as f:
        json.dump(spec, f, indent=2)
    
    print(f"✓ Updated {spec_file} with your preferences")
    
    # Run pipeline
    pipeline = ClaraMasterPipeline(spec_file)
    
    # Check if models exist (to skip training)
    models_exist = any(
        (pipeline.dirs['models'] / f"tinyllama_{dim}").exists()
        for dim in list(spec['personality'].keys()) + list(spec['expertise'].keys())
    )
    
    pipeline.run_complete_pipeline(skip_training=models_exist)


if __name__ == "__main__":
    main()