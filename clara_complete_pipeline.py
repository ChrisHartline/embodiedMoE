"""
Complete Clara Creation Pipeline
Integrates: Data generation, Fine-tuning, Merging, HDC Memory, Monitoring
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

# Import our modules
from clara_monitoring_complete import ClaraMonitor

# Load environment
load_dotenv()


class ClaraCompletePipeline:
    """
    End-to-end pipeline for creating Clara
    """
    
    def __init__(
        self,
        project_name: str = "clara-deng-research",
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ):
        self.project_name = project_name
        self.base_model = base_model
        
        # Setup directories
        self.setup_directories()
        
        # Initialize monitoring
        self.monitor = ClaraMonitor(project_name=project_name)
        
        # Configuration
        self.personality_dimensions = ['warmth', 'formality', 'verbosity', 'encouragement']
        self.domain_areas = ['robotics', 'python_programming', 'teaching']
        
        print("=" * 70)
        print("CLARA COMPLETE PIPELINE - INITIALIZED")
        print("=" * 70)
        print(f"Base model: {base_model}")
        print(f"Project: {project_name}")
        print(f"Personality dimensions: {self.personality_dimensions}")
        print(f"Domain areas: {self.domain_areas}")
    
    def setup_directories(self):
        """Create necessary directories"""
        self.dirs = {
            'data': Path('./data'),
            'models': Path('./models'),
            'configs': Path('./configs'),
            'results': Path('./results'),
            'checkpoints': Path('./checkpoints')
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print("\n✓ Directories created/verified")
    
    # ========================================
    # PHASE 1: DATA GENERATION
    # ========================================
    
    def generate_personality_data(
        self,
        dimension: str,
        n_examples: int = 100,
        use_existing: bool = True
    ) -> Path:
        """
        Generate personality training data
        
        Args:
            dimension: Personality dimension
            n_examples: Number of examples to generate
            use_existing: Use existing data if available
            
        Returns:
            Path to generated data file
        """
        
        print(f"\n{'=' * 70}")
        print(f"PHASE 1: Generate '{dimension}' Training Data")
        print(f"{'=' * 70}")
        
        data_file = self.dirs['data'] / f"{dimension}_training.json"
        
        # Check if data already exists
        if use_existing and data_file.exists():
            print(f"✓ Found existing data: {data_file}")
            with open(data_file) as f:
                existing_data = json.load(f)
            
            print(f"  Examples: {len(existing_data)}")
            
            # Still track it in W&B
            self.monitor.track_data_generation(
                dimension=dimension,
                n_examples=len(existing_data),
                api_cost=0.0,  # Already paid
                examples=existing_data,
                quality_metrics={"source": "existing"}
            )
            
            return data_file
        
        # Generate new data
        print(f"Generating {n_examples} examples for '{dimension}'...")
        print("This will use Claude API (see 2_personality_data.py)")
        
        # Import the data generation code
        try:
            from personality_data_generator import generate_personality_examples, save_examples
            
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                print("⚠️  No ANTHROPIC_API_KEY found!")
                print("   Using mock data for demo...")
                examples = self._generate_mock_data(dimension, n_examples)
                estimated_cost = 0.0
            else:
                examples = generate_personality_examples(
                    dimension=dimension,
                    n_examples=n_examples,
                    api_key=api_key
                )
                estimated_cost = (n_examples * 2) * 0.003 / 1000  # Rough estimate
            
            # Save data
            save_examples(examples, data_file.name)
            
            # Track in W&B
            self.monitor.track_data_generation(
                dimension=dimension,
                n_examples=len(examples),
                api_cost=estimated_cost,
                examples=examples
            )
            
            print(f"✓ Generated {len(examples)} examples")
            print(f"  Cost: ${estimated_cost:.2f}")
            print(f"  Saved to: {data_file}")
            
            return data_file
            
        except ImportError:
            print("⚠️  Using mock data (personality_data_generator not found)")
            examples = self._generate_mock_data(dimension, n_examples)
            
            with open(data_file, 'w') as f:
                json.dump(examples, f, indent=2)
            
            self.monitor.track_data_generation(
                dimension=dimension,
                n_examples=len(examples),
                api_cost=0.0,
                examples=examples,
                quality_metrics={"source": "mock"}
            )
            
            return data_file
    
    def _generate_mock_data(self, dimension: str, n_examples: int) -> List[Dict]:
        """Generate mock data for testing"""
        
        templates = {
            'warmth': [
                ("That is correct.", "Correct.", "That's exactly right! Great job!"),
                ("I can help.", "Sure.", "I'd be delighted to help you!"),
                ("Here's the answer.", "Answer:", "Here's the answer you're looking for!")
            ],
            'formality': [
                ("That is correct.", "Yep, that's right!", "That is indeed correct."),
                ("I can help.", "I can help ya!", "I would be pleased to assist."),
                ("Let's do this.", "Let's go!", "Let us proceed.")
            ],
            'verbosity': [
                ("It works.", "Works.", "This approach functions as intended and should resolve your issue."),
                ("That's wrong.", "Wrong.", "That approach is incorrect and will likely lead to errors."),
                ("Good job.", "Nice.", "Excellent work! You've done a great job on this.")
            ],
            'encouragement': [
                ("You did it.", "Done.", "You did it! That's fantastic progress!"),
                ("Try again.", "Retry.", "Don't give up! You're making great progress, try again!"),
                ("Good.", "OK.", "That's wonderful! Keep up the great work!")
            ]
        }
        
        base_templates = templates.get(dimension, templates['warmth'])
        examples = []
        
        for i in range(n_examples):
            template = base_templates[i % len(base_templates)]
            examples.append({
                'neutral': template[0],
                'low': template[1],
                'high': template[2],
                'dimension': dimension
            })
        
        return examples
    
    def generate_all_personality_data(self, n_examples: int = 100) -> Dict[str, Path]:
        """Generate data for all personality dimensions"""
        
        data_files = {}
        
        for dimension in self.personality_dimensions:
            data_file = self.generate_personality_data(
                dimension=dimension,
                n_examples=n_examples
            )
            data_files[dimension] = data_file
            time.sleep(1)  # Brief pause between dimensions
        
        print(f"\n✓ Generated/verified data for {len(data_files)} dimensions")
        
        return data_files
    
    # ========================================
    # PHASE 2: FINE-TUNING (Colab)
    # ========================================
    
    def prepare_finetuning_instructions(self, dimension: str) -> str:
        """
        Generate instructions for fine-tuning in Colab
        """
        
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: Fine-tune '{dimension}' Model")
        print(f"{'=' * 70}")
        
        data_file = self.dirs['data'] / f"{dimension}_training.json"
        output_dir = self.dirs['models'] / f"tinyllama_{dimension}"
        
        # Setup W&B tracking
        training_config = self.monitor.setup_training_tracking(
            model_name=f"tinyllama_{dimension}",
            dimension=dimension,
            base_model=self.base_model,
            hyperparameters={
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "lora_r": 16,
                "lora_alpha": 32
            }
        )
        
        instructions = f"""
{'=' * 70}
FINE-TUNING INSTRUCTIONS FOR '{dimension.upper()}'
{'=' * 70}

📁 Files needed:
   Data: {data_file}
   Output: {output_dir}

🚀 Run in Google Colab Pro+:
   1. Upload {data_file} to Google Drive
   2. Copy the code from 'clara_colab_finetune.py'
   3. Update these variables:
      - dimension = "{dimension}"
      - data_file = "/content/drive/MyDrive/clara_data/{dimension}_training.json"
      - output_dir = "/content/drive/MyDrive/clara_models/tinyllama_{dimension}"
   
   4. Run all cells
   5. Model will auto-save to Google Drive

⚙️  Training config (already logged in W&B):
   {json.dumps(training_config, indent=2)}

📊 Monitor training:
   W&B will automatically track:
   - Loss curves
   - Learning rate
   - GPU utilization
   - Gradient norms
   
   View at: https://wandb.ai/your-username/{self.project_name}

⏱️  Estimated time:
   - A100: ~30 minutes
   - V100: ~60 minutes
   - T4: ~90 minutes

💾 After training:
   1. Download model from Google Drive
   2. Place in: {output_dir}
   3. Run: pipeline.verify_model("{dimension}")

{'=' * 70}
        """
        
        print(instructions)
        
        # Save instructions to file
        instructions_file = self.dirs['results'] / f"finetune_{dimension}_instructions.txt"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"\n✓ Instructions saved to: {instructions_file}")
        
        self.monitor.finish_run()
        
        return instructions
    
    def prepare_all_finetuning_instructions(self):
        """Generate instructions for all models"""
        
        for dimension in self.personality_dimensions:
            self.prepare_finetuning_instructions(dimension)
            print("\n")
        
        for domain in self.domain_areas:
            self.prepare_finetuning_instructions(domain)
            print("\n")
    
    def verify_model(self, dimension: str) -> bool:
        """Verify that fine-tuned model exists"""
        
        model_dir = self.dirs['models'] / f"tinyllama_{dimension}"
        
        if not model_dir.exists():
            print(f"❌ Model not found: {model_dir}")
            return False
        
        # Check for required files
        required_files = ['config.json', 'tokenizer_config.json']
        for file in required_files:
            if not (model_dir / file).exists():
                print(f"❌ Missing file: {file}")
                return False
        
        print(f"✓ Model verified: {model_dir}")
        return True
    
    # ========================================
    # PHASE 3: MODEL MERGING
    # ========================================
    
    def create_merge_config(
        self,
        merge_name: str,
        personality_weights: Dict[str, float],
        domain_weights: Optional[Dict[str, float]] = None,
        merge_method: str = "linear"
    ) -> Path:
        """
        Create mergekit configuration
        
        Args:
            merge_name: Name for merged model
            personality_weights: Weights for personality dimensions
            domain_weights: Weights for domain models (optional)
            merge_method: Merge method (linear, dare, ties, slerp)
        """
        
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: Create Merge Config - '{merge_name}'")
        print(f"{'=' * 70}")
        
        domain_weights = domain_weights or {}
        
        # Track in W&B
        self.monitor.track_merge_experiment(
            merge_name=merge_name,
            config_path="",  # Will be set after creation
            personality_weights=personality_weights,
            domain_weights=domain_weights,
            merge_method=merge_method
        )
        
        # Build config
        sources = []
        weights = []
        
        # Add personality models
        for dim, weight in personality_weights.items():
            model_path = str(self.dirs['models'] / f"tinyllama_{dim}")
            sources.append({
                "model": model_path,
                "layer_range": [0, 22]
            })
            weights.append(weight)
        
        # Add domain models
        for domain, weight in domain_weights.items():
            model_path = str(self.dirs['models'] / f"tinyllama_{domain}")
            sources.append({
                "model": model_path,
                "layer_range": [0, 22]
            })
            weights.append(weight)
        
        # Create YAML config
        config_yaml = f"""# Clara Merge Configuration: {merge_name}
# Generated: {datetime.now().isoformat()}

merge_method: {merge_method}
base_model: {self.base_model}
dtype: q4_k_m
out_path: {self.dirs['models'] / f'clara_{merge_name}'}
tokenizer_source: base

slices:
  - sources:
"""
        
        for i, source in enumerate(sources):
            config_yaml += f"""      - model: '{source['model']}'
        layer_range: {source['layer_range']}
"""
        
        config_yaml += f"""
weight: {weights}

parameters:
  - name: model.embed_tokens.weight
    method: {merge_method}
    weight: {weights}
  
  - name: lm_head.weight
    method: {merge_method}
    weight: {weights}
"""
        
        # Save config
        config_file = self.dirs['configs'] / f"merge_{merge_name}.yml"
        with open(config_file, 'w') as f:
            f.write(config_yaml)
        
        print(f"\n✓ Merge config created: {config_file}")
        print(f"\n📋 Configuration:")
        print(f"  Method: {merge_method}")
        print(f"  Sources: {len(sources)}")
        print(f"  Personality weights: {personality_weights}")
        if domain_weights:
            print(f"  Domain weights: {domain_weights}")
        
        # Log config as artifact
        self.monitor.log_merge_results(
            merge_name=merge_name,
            output_path=str(self.dirs['models'] / f'clara_{merge_name}'),
            model_size_mb=0.0,  # Not merged yet
            merge_time_seconds=0.0,
            evaluation_metrics={"status": "config_created"}
        )
        
        return config_file
    
    def run_merge(self, config_file: Path) -> Path:
        """
        Execute model merge using mergekit
        
        Args:
            config_file: Path to merge config
            
        Returns:
            Path to merged model
        """
        
        print(f"\n{'=' * 70}")
        print(f"Running Merge: {config_file.name}")
        print(f"{'=' * 70}")
        
        merge_name = config_file.stem.replace('merge_', '')
        output_path = self.dirs['models'] / f'clara_{merge_name}'
        
        # Command to run
        cmd = f"mergekit-yaml {config_file} {output_path} --copy-tokenizer --cuda"
        
        print(f"\n📝 Merge command:")
        print(f"  {cmd}")
        
        print(f"\n⚙️  To run the merge:")
        print(f"  1. Ensure mergekit is installed: pip install mergekit")
        print(f"  2. Run the command above")
        print(f"  3. Or run: pipeline.execute_merge_command('{config_file}')")
        
        print(f"\n⏱️  Estimated time:")
        print(f"  - CPU: 5-10 minutes")
        print(f"  - GPU: 1-2 minutes")
        
        return output_path
    
    def execute_merge_command(self, config_file: Path):
        """Actually execute the merge (requires mergekit installed)"""
        
        import subprocess
        
        merge_name = config_file.stem.replace('merge_', '')
        output_path = self.dirs['models'] / f'clara_{merge_name}'
        
        cmd = [
            "mergekit-yaml",
            str(config_file),
            str(output_path),
            "--copy-tokenizer"
        ]
        
        # Add --cuda if available
        try:
            import torch
            if torch.cuda.is_available():
                cmd.append("--cuda")
        except ImportError:
            pass
        
        print(f"\n🔀 Executing merge...")
        print(f"  Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            merge_time = time.time() - start_time
            
            print(f"\n✓ Merge completed in {merge_time:.1f}s")
            
            # Calculate model size
            model_size_mb = sum(
                f.stat().st_size for f in output_path.rglob('*') if f.is_file()
            ) / (1024 * 1024)
            
            # Log results
            self.monitor.log_merge_results(
                merge_name=merge_name,
                output_path=str(output_path),
                model_size_mb=model_size_mb,
                merge_time_seconds=merge_time
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Merge failed: {e}")
            print(f"  Output: {e.output}")
            return None
        except FileNotFoundError:
            print("\n❌ mergekit not found!")
            print("  Install with: pip install mergekit")
            return None
    
    # ========================================
    # PHASE 4: EVALUATION
    # ========================================
    
    def evaluate_merged_model(
        self,
        model_path: Path,
        test_prompts: Optional[List[str]] = None
    ):
        """
        Evaluate merged Clara model
        """
        
        print(f"\n{'=' * 70}")
        print(f"PHASE 4: Evaluate Merged Model")
        print(f"{'=' * 70}")
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        test_prompts = test_prompts or [
            "Can you help me with this problem?",
            "I'm stuck on async programming.",
            "How do I grasp objects with a robot?",
            "Thanks for your help!",
            "Explain this concept to me."
        ]
        
        print(f"\n📝 Test prompts: {len(test_prompts)}")
        
        # Try to load model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            print(f"✓ Model loaded on {device}")
            
            # Generate function
            def generate(prompt):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=100,
                        temperature=0.7,
                        do_sample=True
                    )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Evaluate personality
            metrics = self.monitor.evaluate_personality_consistency(
                model_path=str(model_path),
                dimension="overall",
                test_prompts=test_prompts,
                expected_traits={"warmth": 0.8, "helpfulness": 0.9},
                generate_fn=generate
            )
            
            print(f"\n✓ Evaluation complete")
            print(f"  Metrics: {metrics}")
            
        except ImportError:
            print("⚠️  transformers not installed, skipping actual model loading")
            print("  Install with: pip install transformers torch")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    # ========================================
    # COMPLETE PIPELINE RUNNER
    # ========================================
    
    def run_complete_pipeline(
        self,
        n_examples: int = 100,
        skip_data_generation: bool = False,
        skip_finetuning: bool = True,  # Usually done in Colab
        skip_merging: bool = False
    ):
        """
        Run the complete pipeline end-to-end
        
        Args:
            n_examples: Number of training examples per dimension
            skip_data_generation: Skip if data already exists
            skip_finetuning: Skip fine-tuning step (do in Colab)
            skip_merging: Skip model merging
        """
        
        print("\n" + "=" * 70)
        print("CLARA COMPLETE PIPELINE - START")
        print("=" * 70)
        
        # Phase 1: Data Generation
        if not skip_data_generation:
            print("\n📊 PHASE 1: Data Generation")
            self.generate_all_personality_data(n_examples=n_examples)
        else:
            print("\n⏭️  Skipping data generation (using existing)")
        
        # Phase 2: Fine-tuning Instructions
        if not skip_finetuning:
            print("\n🎓 PHASE 2: Fine-tuning (Colab)")
            self.prepare_all_finetuning_instructions()
            print("\n⚠️  Complete fine-tuning in Colab, then continue pipeline")
            return
        else:
            print("\n⏭️  Skipping fine-tuning instructions")
        
        # Phase 3: Model Merging
        if not skip_merging:
            print("\n🔀 PHASE 3: Model Merging")
            
            # Create merge config
            config_file = self.create_merge_config(
                merge_name="v1_balanced",
                personality_weights={
                    "warmth": 0.3,
                    "formality": 0.1,
                    "encouragement": 0.3
                },
                domain_weights={
                    "robotics": 0.2,
                    "python_programming": 0.1
                },
                merge_method="linear"
            )
            
            # Show merge instructions
            self.run_merge(config_file)
            
            print("\n⚠️  Run merge command, then evaluate")
        else:
            print("\n⏭️  Skipping merging")
        
        print("\n" + "=" * 70)
        print("PIPELINE STAGE COMPLETE")
        print("=" * 70)
        
        print("""
Next steps:
1. If data generated: Review in ./data/
2. If training needed: Fine-tune models in Colab
3. If models ready: Run merge with mergekit
4. After merge: Evaluate with pipeline.evaluate_merged_model()

View progress: https://wandb.ai/your-username/clara-deng-research
        """)


def main():
    """
    Main entry point with interactive menu
    """
    
    pipeline = ClaraCompletePipeline()
    
    print("\n" + "=" * 70)
    print("CLARA PIPELINE - INTERACTIVE MODE")
    print("=" * 70)
    
    print("""
What would you like to do?

1. Generate personality training data
2. Generate ALL training data
3. Prepare fine-tuning instructions
4. Verify fine-tuned models
5. Create merge configuration
6. Run complete pipeline (data + instructions)
7. Evaluate merged model
8. Exit

(You can also share your existing data/code for integration)
    """)
    
    choice = input("Enter choice (1-8): ").strip()
    
    if choice == "1":
        dimension = input("Enter dimension (warmth/formality/verbosity/encouragement): ").strip()
        n_examples = int(input("Number of examples (default 100): ") or "100")
        pipeline.generate_personality_data(dimension, n_examples, use_existing=False)
    
    elif choice == "2":
        n_examples = int(input("Number of examples per dimension (default 100): ") or "100")
        pipeline.generate_all_personality_data(n_examples)
    
    elif choice == "3":
        pipeline.prepare_all_finetuning_instructions()
    
    elif choice == "4":
        dimension = input("Enter dimension to verify: ").strip()
        pipeline.verify_model(dimension)
    
    elif choice == "5":
        merge_name = input("Merge name (e.g., v1_balanced): ").strip()
        print("Enter personality weights (warmth, formality, etc.):")
        # Simplified - in practice would ask for each weight
        config = pipeline.create_merge_config(
            merge_name=merge_name,
            personality_weights={"warmth": 0.4, "encouragement": 0.4},
            domain_weights={"robotics": 0.2}
        )
        print(f"\nConfig created: {config}")
    
    elif choice == "6":
        pipeline.run_complete_pipeline(
            n_examples=100,
            skip_data_generation=False,
            skip_finetuning=True,
            skip_merging=True
        )
    
    elif choice == "7":
        model_path = input("Enter path to merged model: ").strip()
        pipeline.evaluate_merged_model(Path(model_path))
    
    elif choice == "8":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()