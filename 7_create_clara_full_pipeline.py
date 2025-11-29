"""
Complete Pipeline: Create Clara with Personality + Domain Knowledge
Using mergekit to combine:
- Personality fine-tunes (warmth, formality, etc.)
- Domain knowledge fine-tunes (robotics, programming, etc.)
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path

class ClaraModelPipeline:
    """
    Complete pipeline for creating Clara through model merging
    """
    
    def __init__(self, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.base_model = base_model
        self.models_dir = Path("./models")
        self.data_dir = Path("./data")
        self.configs_dir = Path("./configs")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("CLARA CREATION PIPELINE")
        print("=" * 70)
        print(f"Base model: {base_model}")
    
    def step1_generate_personality_data(self, dimensions: List[str], examples_per_dim: int = 100):
        """
        Step 1: Generate training data for personality dimensions
        """
        print("\n" + "=" * 70)
        print("STEP 1: GENERATE PERSONALITY TRAINING DATA")
        print("=" * 70)
        
        # This would use the code from 2_personality_data.py
        print(f"\nGenerating data for personality dimensions: {dimensions}")
        print(f"Examples per dimension: {examples_per_dim}")
        
        for dim in dimensions:
            output_file = self.data_dir / f"{dim}_training.json"
            print(f"  → {dim}: {output_file}")
            print(f"     Command: python 2_personality_data.py --dimension {dim} --n {examples_per_dim}")
        
        print("\n💡 This step uses Claude API to generate transformation pairs")
        print("   Example for 'warmth':")
        print("     Neutral:  'That is correct.'")
        print("     Low:      'Correct.'")
        print("     High:     'That's exactly right! Great job!'")
    
    def step2_generate_domain_data(self, domains: List[str]):
        """
        Step 2: Generate/collect domain-specific knowledge data
        """
        print("\n" + "=" * 70)
        print("STEP 2: GENERATE DOMAIN KNOWLEDGE DATA")
        print("=" * 70)
        
        print(f"\nDomain specializations: {domains}")
        
        for domain in domains:
            output_file = self.data_dir / f"{domain}_knowledge.json"
            print(f"\n  → {domain}:")
            print(f"     Output: {output_file}")
            
            if domain == "robotics":
                print("     Sources: Robotics papers, ROS documentation, manipulation tutorials")
                print("     Focus: Grasping, navigation, object detection, safety")
                
            elif domain == "python_programming":
                print("     Sources: Python docs, async tutorials, debugging guides")
                print("     Focus: Async/await, error handling, best practices")
                
            elif domain == "teaching":
                print("     Sources: Educational content, tutoring dialogues")
                print("     Focus: Explaining concepts, using analogies, patience")
                
            elif domain == "medical":
                print("     Sources: Medical textbooks, clinical notes (if available)")
                print("     Focus: Medical terminology, patient interaction")
        
        print("\n💡 Domain data can come from:")
        print("   - Existing datasets (HuggingFace, GitHub)")
        print("   - Curated documents (textbooks, manuals)")
        print("   - Synthetic data (Claude API generating Q&A pairs)")
    
    def step3_finetune_models(self, personality_dims: List[str], domains: List[str]):
        """
        Step 3: Fine-tune separate models for each dimension/domain
        """
        print("\n" + "=" * 70)
        print("STEP 3: FINE-TUNE SPECIALIST MODELS")
        print("=" * 70)
        
        all_models = {}
        
        print("\nPersonality Models:")
        for dim in personality_dims:
            model_path = self.models_dir / f"tinyllama_{dim}"
            all_models[f"personality_{dim}"] = str(model_path)
            
            print(f"\n  {dim}:")
            print(f"    Input:  {self.data_dir / f'{dim}_training.json'}")
            print(f"    Output: {model_path}")
            print(f"    Command:")
            print(f"""
    python -m transformers.examples.pytorch.language-modeling.run_clm \\
        --model_name_or_path {self.base_model} \\
        --train_file {self.data_dir / f'{dim}_training.json'} \\
        --output_dir {model_path} \\
        --num_train_epochs 3 \\
        --per_device_train_batch_size 4 \\
        --save_steps 500 \\
        --learning_rate 5e-5
            """)
        
        print("\nDomain Knowledge Models:")
        for domain in domains:
            model_path = self.models_dir / f"tinyllama_{domain}"
            all_models[f"domain_{domain}"] = str(model_path)
            
            print(f"\n  {domain}:")
            print(f"    Input:  {self.data_dir / f'{domain}_knowledge.json'}")
            print(f"    Output: {model_path}")
            print(f"    Training: Same command as above, different data")
        
        print(f"\n💡 This creates {len(all_models)} specialist models")
        print("   Each one is TinyLlama fine-tuned on specific data")
        print("   Training time: ~2-4 hours each on single GPU")
        
        return all_models
    
    def step4_create_merge_config(
        self,
        personality_weights: Dict[str, float],
        domain_weights: Dict[str, float],
        all_models: Dict[str, str]
    ):
        """
        Step 4: Create mergekit configuration
        """
        print("\n" + "=" * 70)
        print("STEP 4: CREATE MERGE CONFIGURATION")
        print("=" * 70)
        
        print("\nClara's Configuration:")
        print("\nPersonality Weights:")
        for dim, weight in personality_weights.items():
            print(f"  {dim}: {weight}")
        
        print("\nDomain Knowledge Weights:")
        for domain, weight in domain_weights.items():
            print(f"  {domain}: {weight}")
        
        # Create merge config
        config = {
            "merge_method": "linear",
            "slices": [
                {
                    "sources": []
                }
            ],
            "dtype": "float16",
            "out_dtype": "float16"
        }
        
        # Add personality models
        for dim, weight in personality_weights.items():
            model_key = f"personality_{dim}"
            if model_key in all_models:
                config["slices"][0]["sources"].append({
                    "model": all_models[model_key],
                    "layer_range": [0, 22],
                    "parameters": {
                        "weight": weight
                    }
                })
        
        # Add domain models
        for domain, weight in domain_weights.items():
            model_key = f"domain_{domain}"
            if model_key in all_models:
                config["slices"][0]["sources"].append({
                    "model": all_models[model_key],
                    "layer_range": [0, 22],
                    "parameters": {
                        "weight": weight
                    }
                })
        
        # Save config
        config_path = self.configs_dir / "clara_full_merge.yaml"
        
        # Convert to YAML format (mergekit uses YAML)
        yaml_content = self._config_to_yaml(config)
        
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Config saved to: {config_path}")
        print("\nMerge sources:")
        for i, src in enumerate(config["slices"][0]["sources"], 1):
            model_name = Path(src["model"]).name
            weight = src["parameters"]["weight"]
            print(f"  {i}. {model_name}: {weight}")
        
        return config_path
    
    def _config_to_yaml(self, config: Dict) -> str:
        """Convert config dict to YAML string"""
        yaml_lines = [
            f"merge_method: {config['merge_method']}",
            f"dtype: {config['dtype']}",
            f"out_dtype: {config['out_dtype']}",
            "slices:"
        ]
        
        for slice_config in config["slices"]:
            yaml_lines.append("  - sources:")
            for src in slice_config["sources"]:
                yaml_lines.append(f"      - model: {src['model']}")
                yaml_lines.append(f"        layer_range: {src['layer_range']}")
                yaml_lines.append(f"        parameters:")
                yaml_lines.append(f"          weight: {src['parameters']['weight']}")
        
        return "\n".join(yaml_lines)
    
    def step5_merge_models(self, config_path: Path):
        """
        Step 5: Run mergekit to create Clara
        """
        print("\n" + "=" * 70)
        print("STEP 5: MERGE MODELS TO CREATE CLARA")
        print("=" * 70)
        
        output_path = self.models_dir / "clara_merged"
        
        print(f"\nInput:  {config_path}")
        print(f"Output: {output_path}")
        
        print("\nMerge command:")
        print(f"  mergekit-yaml {config_path} {output_path} --copy-tokenizer")
        
        print("\n💡 This combines all specialist models into one!")
        print("   Merging time: ~5-10 minutes")
        print("   Output: Single TinyLlama with Clara's personality + knowledge")
    
    def step6_deploy_with_hdc(self, merged_model_path: Path):
        """
        Step 6: Deploy Clara with HDC memory
        """
        print("\n" + "=" * 70)
        print("STEP 6: DEPLOY CLARA WITH HDC MEMORY")
        print("=" * 70)
        
        print("\nFinal Architecture:")
        print("""
        ┌─────────────────────────────────────────┐
        │              CLARA                      │
        ├─────────────────────────────────────────┤
        │                                         │
        │  Merged TinyLlama Model                 │
        │  ├─ Personality (from merge)            │
        │  │  ├─ Warmth: 80%                      │
        │  │  ├─ Formality: 30%                   │
        │  │  └─ Encouragement: 90%               │
        │  │                                      │
        │  ├─ Knowledge (from merge)              │
        │  │  ├─ Robotics: Expert                 │
        │  │  ├─ Python: Advanced                 │
        │  │  └─ Teaching: Strong                 │
        │  │                                      │
        │  └─ Generation: Creates responses       │
        │                                         │
        │  HDC Memory System                      │
        │  ├─ Episodic: Recent experiences        │
        │  ├─ User Profile: Learned preferences   │
        │  └─ Retrieval: Context for generation   │
        │                                         │
        └─────────────────────────────────────────┘
        """)
        
        print(f"Model location: {merged_model_path}")
        print("\nIntegration code:")
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from hdc_with_embeddings import HDCMemoryWithEmbeddings

# Load merged Clara model
clara_model = AutoModelForCausalLM.from_pretrained("{merged_model_path}")
clara_tokenizer = AutoTokenizer.from_pretrained("{merged_model_path}")

# Initialize HDC memory (using Clara model for embeddings!)
clara_memory = HDCMemoryWithEmbeddings(
    encoder_model=clara_model,  # Use Clara for encoding too!
    tokenizer=clara_tokenizer
)

# Clara is ready!
def clara_respond(user_input):
    # Retrieve context
    context = clara_memory.query_similar(user_input)
    
    # Generate with Clara's personality + knowledge
    response = generate_with_context(user_input, context)
    
    # Store experience
    clara_memory.store_experience(user_input, response)
    
    return response
        """)
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline (demonstration)
        """
        
        # Define Clara's specifications
        personality_dims = ['warmth', 'formality', 'verbosity', 'encouragement']
        domains = ['robotics', 'python_programming', 'teaching']
        
        # Clara's personality profile
        personality_weights = {
            'warmth': 0.8,
            'formality': 0.3,
            'verbosity': 0.6,
            'encouragement': 0.9
        }
        
        # Clara's knowledge profile
        domain_weights = {
            'robotics': 0.8,          # Expert in robotics
            'python_programming': 0.6, # Advanced Python knowledge
            'teaching': 0.7           # Strong teaching ability
        }
        
        # Run each step
        self.step1_generate_personality_data(personality_dims, examples_per_dim=1000)
        
        self.step2_generate_domain_data(domains)
        
        all_models = self.step3_finetune_models(personality_dims, domains)
        
        config_path = self.step4_create_merge_config(
            personality_weights,
            domain_weights,
            all_models
        )
        
        self.step5_merge_models(config_path)
        
        merged_path = self.models_dir / "clara_merged"
        self.step6_deploy_with_hdc(merged_path)
        
        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        
        print("\n📊 What You've Created:")
        print(f"  - {len(personality_dims)} personality models")
        print(f"  - {len(domains)} domain knowledge models")
        print(f"  - 1 merged Clara model with unique personality + expertise")
        print(f"  - HDC memory system for experience storage")
        
        print("\n🎯 Clara's Capabilities:")
        print("  ✓ Warm, casual, encouraging communication style")
        print("  ✓ Expert robotics knowledge (grasping, navigation)")
        print("  ✓ Advanced Python programming (especially async)")
        print("  ✓ Strong teaching ability (analogies, examples)")
        print("  ✓ Episodic memory via HDC")
        print("  ✓ User preference learning")
        
        print("\n💾 Total Size:")
        print("  - Merged model: ~2.2GB (TinyLlama 1.1B)")
        print("  - HDC memory: ~200KB")
        print("  - Total: ~2.2GB (fits on edge devices!)")
        
        print("\n⚡ Performance:")
        print("  - Inference: 10-50ms per token (on RTX 2080)")
        print("  - Memory retrieval: <5ms")
        print("  - Runs entirely on-device!")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR YOUR D.ENG")
        print("=" * 70)
        
        print("""
1. PHASE 1: Data Generation (Week 1-2)
   - Generate personality datasets with Claude API
   - Curate domain knowledge from existing sources
   - Validate data quality

2. PHASE 2: Model Training (Week 3-6)
   - Fine-tune 7 specialist models (4 personality + 3 domain)
   - Each takes 2-4 hours on your RTX 2080
   - Validate each model individually

3. PHASE 3: Merging & Testing (Week 7-8)
   - Experiment with different merge weights
   - Test personality consistency
   - Benchmark against baselines

4. PHASE 4: Integration (Week 9-10)
   - Integrate with HDC memory
   - Deploy on robot platform
   - Test in real scenarios

5. PHASE 5: Evaluation (Week 11-12)
   - User studies on personality consistency
   - Performance benchmarks
   - Write up results for D.Eng

📝 Research Contributions:
   - Novel use of model merging for personality
   - Hybrid neural-symbolic architecture (merged LM + HDC)
   - On-device personal AI without cloud
   - Demonstrable personality emergence from structure
        """)


if __name__ == "__main__":
    pipeline = ClaraModelPipeline()
    pipeline.run_full_pipeline()