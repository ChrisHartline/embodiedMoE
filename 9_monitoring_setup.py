"""
Complete Clara Monitoring System with Weights & Biases Integration
Tracks: data generation, fine-tuning, merging, evaluation, HDC memory
"""

import wandb
import os
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ClaraMonitor:
    """
    Complete monitoring system for Clara development pipeline
    Integrates with Weights & Biases for comprehensive tracking
    """
    
    def __init__(
        self,
        project_name: str = "clara-deng-research",
        entity: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Clara monitoring system
        
        Args:
            project_name: W&B project name
            entity: W&B username/team (optional)
            api_key: W&B API key (or uses WB env variable)
        """
        self.project_name = project_name
        self.entity = entity
        
        # Get API key from parameter, env variable, or prompt
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        elif 'WB' in os.environ:
            os.environ['WANDB_API_KEY'] = os.environ['WB']
        
        # Initialize W&B (login)
        try:
            wandb.login(key=os.environ.get('WANDB_API_KEY'))
            print("✓ Logged into Weights & Biases")
        except Exception as e:
            print(f"⚠️  W&B login issue: {e}")
            print("Run: wandb login")
        
        self.current_run = None
        
        print("=" * 70)
        print("CLARA MONITORING SYSTEM - INITIALIZED")
        print("=" * 70)
        print(f"Project: {project_name}")
        if entity:
            print(f"Entity: {entity}")
        print(f"Dashboard: https://wandb.ai/{entity or 'your-username'}/{project_name}")
    
    def start_run(
        self,
        run_name: str,
        job_type: str,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Start a new W&B run
        
        Args:
            run_name: Name for this run
            job_type: Type of job (data_generation, training, evaluation, etc.)
            config: Configuration dictionary
            tags: List of tags for filtering
        """
        self.current_run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            job_type=job_type,
            config=config or {},
            tags=tags or [],
            reinit=True
        )
        
        print(f"\n🚀 Started W&B run: {run_name}")
        print(f"   Job type: {job_type}")
        print(f"   URL: {self.current_run.url}")
        
        return self.current_run
    
    def finish_run(self):
        """Finish current W&B run"""
        if self.current_run:
            wandb.finish()
            self.current_run = None
            print("✓ Run finished\n")
    
    # ========================================
    # PHASE 1: DATA GENERATION TRACKING
    # ========================================
    
    def track_data_generation(
        self,
        dimension: str,
        n_examples: int,
        api_cost: float,
        examples: List[Dict],
        quality_metrics: Optional[Dict] = None
    ):
        """
        Track personality/domain data generation
        
        Args:
            dimension: Personality dimension or domain (e.g., 'warmth', 'robotics')
            n_examples: Number of examples generated
            api_cost: Cost in USD for API calls
            examples: List of generated examples
            quality_metrics: Optional quality assessment metrics
        """
        
        run = self.start_run(
            run_name=f"data-gen-{dimension}",
            job_type="data_generation",
            config={
                "dimension": dimension,
                "n_examples": n_examples,
                "api_cost_usd": api_cost,
                "timestamp": datetime.now().isoformat()
            },
            tags=["data-generation", dimension]
        )
        
        # Calculate automatic quality metrics
        if examples:
            auto_metrics = self._calculate_data_quality(examples)
        else:
            auto_metrics = {}
        
        # Merge with provided metrics
        all_metrics = {**auto_metrics, **(quality_metrics or {})}
        
        # Log metrics
        wandb.log({
            "examples_generated": n_examples,
            "api_cost_usd": api_cost,
            **all_metrics
        })
        
        # Create dataset artifact
        artifact = wandb.Artifact(
            name=f"{dimension}_training_data",
            type="dataset",
            description=f"Training data for {dimension}",
            metadata={
                "n_examples": n_examples,
                "dimension": dimension,
                "quality_metrics": all_metrics
            }
        )
        
        # Save examples to file and add to artifact
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / f"{dimension}_training.json"
        
        with open(data_file, 'w') as f:
            json.dump(examples, f, indent=2)
        
        artifact.add_file(str(data_file))
        wandb.log_artifact(artifact)
        
        # Create examples table
        table = wandb.Table(
            columns=["neutral", "low", "high", "low_length", "high_length"]
        )
        
        for ex in examples[:20]:  # First 20 examples
            table.add_data(
                ex.get('neutral', ''),
                ex.get('low', ''),
                ex.get('high', ''),
                len(ex.get('low', '').split()),
                len(ex.get('high', '').split())
            )
        
        wandb.log({"example_samples": table})
        
        print(f"\n✓ Tracked data generation for '{dimension}'")
        print(f"  Examples: {n_examples}")
        print(f"  API Cost: ${api_cost:.2f}")
        print(f"  Quality metrics: {all_metrics}")
        
        self.finish_run()
        
        return all_metrics
    
    def _calculate_data_quality(self, examples: List[Dict]) -> Dict[str, float]:
        """Calculate automatic quality metrics from examples"""
        
        if not examples:
            return {}
        
        low_lengths = []
        high_lengths = []
        neutral_lengths = []
        
        for ex in examples:
            if 'low' in ex:
                low_lengths.append(len(ex['low'].split()))
            if 'high' in ex:
                high_lengths.append(len(ex['high'].split()))
            if 'neutral' in ex:
                neutral_lengths.append(len(ex['neutral'].split()))
        
        metrics = {}
        
        if low_lengths:
            metrics['avg_length_low'] = np.mean(low_lengths)
            metrics['std_length_low'] = np.std(low_lengths)
        
        if high_lengths:
            metrics['avg_length_high'] = np.mean(high_lengths)
            metrics['std_length_high'] = np.std(high_lengths)
        
        if neutral_lengths:
            metrics['avg_length_neutral'] = np.mean(neutral_lengths)
        
        # Calculate diversity (unique words / total words)
        if examples:
            all_text = ' '.join([
                ex.get('low', '') + ' ' + ex.get('high', '') 
                for ex in examples
            ])
            words = all_text.lower().split()
            if words:
                metrics['vocabulary_diversity'] = len(set(words)) / len(words)
        
        return metrics
    
    # ========================================
    # PHASE 2: FINE-TUNING TRACKING
    # ========================================
    
    def setup_training_tracking(
        self,
        model_name: str,
        dimension: str,
        base_model: str,
        hyperparameters: Dict
    ) -> Dict:
        """
        Setup tracking for fine-tuning (to be used with HuggingFace Trainer)
        
        Args:
            model_name: Name of output model
            dimension: Personality dimension being trained
            base_model: Base model identifier
            hyperparameters: Training hyperparameters
            
        Returns:
            Dictionary with W&B config for Trainer
        """
        
        run = self.start_run(
            run_name=f"train-{dimension}",
            job_type="training",
            config={
                "model_name": model_name,
                "dimension": dimension,
                "base_model": base_model,
                **hyperparameters
            },
            tags=["training", dimension, base_model.split('/')[-1]]
        )
        
        print(f"\n🔧 Fine-tuning tracking configured for '{dimension}'")
        print("\n  HuggingFace Trainer Integration:")
        print("  Add to TrainingArguments:")
        print("    report_to='wandb'")
        print("    run_name=f'train-{dimension}'")
        
        # Return config for HuggingFace integration
        return {
            "report_to": "wandb",
            "run_name": f"train-{dimension}",
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 100
        }
    
    def log_training_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Manually log training metrics (if not using HuggingFace Trainer)
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (optional)
        """
        wandb.log(metrics, step=step)
    
    # ========================================
    # PHASE 3: PERSONALITY EVALUATION
    # ========================================
    
    def evaluate_personality_consistency(
        self,
        model_path: str,
        dimension: str,
        test_prompts: List[str],
        expected_traits: Dict[str, float],
        generate_fn: callable
    ):
        """
        Evaluate personality consistency of a model
        
        Args:
            model_path: Path to model
            dimension: Personality dimension
            test_prompts: List of test prompts
            expected_traits: Expected trait scores (e.g., {'warmth': 0.8})
            generate_fn: Function that takes prompt and returns response
        """
        
        run = self.start_run(
            run_name=f"eval-personality-{dimension}",
            job_type="evaluation",
            config={
                "model_path": model_path,
                "dimension": dimension,
                "n_test_prompts": len(test_prompts),
                "expected_traits": expected_traits
            },
            tags=["evaluation", "personality", dimension]
        )
        
        print(f"\n📊 Evaluating personality consistency: {dimension}")
        print(f"  Test prompts: {len(test_prompts)}")
        
        # Generate responses
        responses = []
        for prompt in test_prompts:
            response = generate_fn(prompt)
            responses.append({
                'prompt': prompt,
                'response': response
            })
        
        # Analyze responses
        trait_scores = self._analyze_personality_traits(responses, dimension)
        
        # Calculate consistency
        consistency_score = self._calculate_consistency(trait_scores)
        
        # Log metrics
        metrics = {
            "consistency_score": consistency_score,
            f"{dimension}_score": np.mean(list(trait_scores.values())),
            "response_count": len(responses)
        }
        
        wandb.log(metrics)
        
        # Create evaluation table
        eval_table = wandb.Table(
            columns=["prompt", "response", f"{dimension}_detected", "length"]
        )
        
        for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
            eval_table.add_data(
                prompt,
                response['response'][:100] + "..." if len(response['response']) > 100 else response['response'],
                trait_scores.get(i, 0.5),
                len(response['response'])
            )
        
        wandb.log({"evaluation_samples": eval_table})
        
        # Log distribution plot
        if trait_scores:
            wandb.log({
                "trait_score_distribution": wandb.Histogram(list(trait_scores.values()))
            })
        
        print(f"\n  ✓ Consistency score: {consistency_score:.3f}")
        print(f"  ✓ Average {dimension} score: {metrics[f'{dimension}_score']:.3f}")
        
        self.finish_run()
        
        return metrics
    
    def _analyze_personality_traits(self, responses: List[Dict], dimension: str) -> Dict[int, float]:
        """
        Analyze personality traits in responses (simplified heuristic)
        In practice, you'd use a trained classifier or more sophisticated analysis
        """
        
        trait_scores = {}
        
        # Simple heuristics (replace with actual classifier)
        warmth_indicators = ['!', 'happy', 'glad', 'love', 'wonderful', 'great']
        formal_indicators = ['therefore', 'however', 'consequently', 'furthermore']
        
        for i, resp in enumerate(responses):
            text = resp['response'].lower()
            
            if dimension == 'warmth':
                score = sum(1 for indicator in warmth_indicators if indicator in text)
                trait_scores[i] = min(1.0, score / 3.0)  # Normalize
            
            elif dimension == 'formality':
                score = sum(1 for indicator in formal_indicators if indicator in text)
                trait_scores[i] = min(1.0, score / 2.0)
            
            else:
                # Default: random for demonstration
                trait_scores[i] = 0.5
        
        return trait_scores
    
    def _calculate_consistency(self, trait_scores: Dict[int, float]) -> float:
        """Calculate consistency as inverse of standard deviation"""
        if not trait_scores:
            return 0.0
        
        values = list(trait_scores.values())
        return 1.0 - min(1.0, np.std(values))
    
    # ========================================
    # PHASE 4: MERGE EXPERIMENT TRACKING
    # ========================================
    
    def track_merge_experiment(
        self,
        merge_name: str,
        config_path: str,
        personality_weights: Dict[str, float],
        domain_weights: Dict[str, float],
        merge_method: str = "linear"
    ):
        """
        Track model merging experiment
        
        Args:
            merge_name: Name for this merge
            config_path: Path to merge config file
            personality_weights: Personality dimension weights
            domain_weights: Domain knowledge weights
            merge_method: Merge method used
        """
        
        run = self.start_run(
            run_name=f"merge-{merge_name}",
            job_type="merge",
            config={
                "merge_name": merge_name,
                "merge_method": merge_method,
                "personality_weights": personality_weights,
                "domain_weights": domain_weights,
                "total_source_models": len(personality_weights) + len(domain_weights)
            },
            tags=["merge", merge_method, merge_name]
        )
        
        print(f"\n🔀 Tracking merge experiment: {merge_name}")
        print(f"  Method: {merge_method}")
        print(f"  Personality weights: {personality_weights}")
        print(f"  Domain weights: {domain_weights}")
        
        # Log merge configuration as artifact
        config_artifact = wandb.Artifact(
            name=f"merge_config_{merge_name}",
            type="config",
            description=f"Merge configuration for {merge_name}"
        )
        
        if Path(config_path).exists():
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)
        
        # Log weight distributions
        all_weights = {**personality_weights, **domain_weights}
        
        # Create bar chart of weights
        weight_data = [[k, v] for k, v in all_weights.items()]
        weight_table = wandb.Table(data=weight_data, columns=["model", "weight"])
        
        wandb.log({
            "weight_distribution": wandb.plot.bar(
                weight_table,
                "model",
                "weight",
                title="Model Weights in Merge"
            )
        })
        
        return run
    
    def log_merge_results(
        self,
        merge_name: str,
        output_path: str,
        model_size_mb: float,
        merge_time_seconds: float,
        evaluation_metrics: Optional[Dict] = None
    ):
        """
        Log results after merge completes
        
        Args:
            merge_name: Name of merge
            output_path: Path to merged model
            model_size_mb: Size of merged model in MB
            merge_time_seconds: Time taken to merge
            evaluation_metrics: Optional evaluation results
        """
        
        metrics = {
            "model_size_mb": model_size_mb,
            "merge_time_seconds": merge_time_seconds,
            "merge_time_minutes": merge_time_seconds / 60
        }
        
        if evaluation_metrics:
            metrics.update(evaluation_metrics)
        
        wandb.log(metrics)
        
        # Save merged model as artifact
        model_artifact = wandb.Artifact(
            name=f"clara_merged_{merge_name}",
            type="model",
            description=f"Merged Clara model: {merge_name}",
            metadata={
                "size_mb": model_size_mb,
                "merge_time": merge_time_seconds
            }
        )
        
        # Add config files (not the huge model files)
        output_dir = Path(output_path)
        if output_dir.exists():
            for config_file in ["config.json", "tokenizer_config.json"]:
                file_path = output_dir / config_file
                if file_path.exists():
                    model_artifact.add_file(str(file_path))
        
        wandb.log_artifact(model_artifact)
        
        print(f"\n  ✓ Logged merge results")
        print(f"    Size: {model_size_mb:.1f} MB")
        print(f"    Time: {merge_time_seconds:.1f}s")
        
        self.finish_run()
    
    # ========================================
    # PHASE 5: HDC MEMORY EVALUATION
    # ========================================
    
    def evaluate_hdc_memory(
        self,
        memory_system,
        test_queries: List[Dict],
        ground_truth: List[str]
    ):
        """
        Evaluate HDC memory system performance
        
        Args:
            memory_system: HDC memory system instance
            test_queries: List of test queries (with expected results)
            ground_truth: List of expected top matches
        """
        
        run = self.start_run(
            run_name="eval-hdc-memory",
            job_type="memory_evaluation",
            config={
                "n_test_queries": len(test_queries),
                "hdc_dimensions": getattr(memory_system, 'hdc_dimensions', 10000)
            },
            tags=["evaluation", "hdc-memory"]
        )
        
        print("\n🧠 Evaluating HDC Memory System")
        print(f"  Test queries: {len(test_queries)}")
        
        results = []
        retrieval_times = []
        accuracies = []
        
        for i, (query_dict, expected) in enumerate(zip(test_queries, ground_truth)):
            query = query_dict['query']
            
            # Time the retrieval
            start_time = time.time()
            retrieved = memory_system.query_similar(query, top_k=3)
            retrieval_time = (time.time() - start_time) * 1000  # ms
            
            retrieval_times.append(retrieval_time)
            
            # Check if expected result is in top 3
            is_correct = any(expected in r[0] for r in retrieved) if retrieved else False
            accuracies.append(1.0 if is_correct else 0.0)
            
            results.append({
                'query': query,
                'expected': expected,
                'retrieved_top1': retrieved[0][0] if retrieved else None,
                'similarity': retrieved[0][1] if retrieved else 0.0,
                'correct': is_correct,
                'time_ms': retrieval_time
            })
        
        # Calculate metrics
        metrics = {
            "retrieval_accuracy": np.mean(accuracies),
            "avg_retrieval_time_ms": np.mean(retrieval_times),
            "median_retrieval_time_ms": np.median(retrieval_times),
            "memory_size_kb": getattr(memory_system, 'hdc_dimensions', 10000) * len(memory_system.memories) * 4 / 1024 if hasattr(memory_system, 'memories') else 0
        }
        
        wandb.log(metrics)
        
        # Create results table
        results_table = wandb.Table(
            columns=["query", "expected", "retrieved", "similarity", "correct", "time_ms"]
        )
        
        for r in results[:20]:  # First 20
            results_table.add_data(
                r['query'][:50],
                r['expected'][:50] if r['expected'] else "N/A",
                r['retrieved_top1'][:50] if r['retrieved_top1'] else "N/A",
                r['similarity'],
                r['correct'],
                r['time_ms']
            )
        
        wandb.log({"retrieval_results": results_table})
        
        # Log timing distribution
        wandb.log({
            "retrieval_time_distribution": wandb.Histogram(retrieval_times)
        })
        
        print(f"\n  ✓ Accuracy: {metrics['retrieval_accuracy']:.2%}")
        print(f"  ✓ Avg retrieval time: {metrics['avg_retrieval_time_ms']:.2f}ms")
        print(f"  ✓ Memory size: {metrics['memory_size_kb']:.1f}KB")
        
        self.finish_run()
        
        return metrics
    
    # ========================================
    # DASHBOARD & COMPARISON
    # ========================================
    
    def create_comparison_report(
        self,
        experiment_name: str,
        experiments: List[Dict[str, Any]]
    ):
        """
        Create comparison report across multiple experiments
        
        Args:
            experiment_name: Name for this comparison
            experiments: List of experiment results to compare
        """
        
        run = self.start_run(
            run_name=f"compare-{experiment_name}",
            job_type="comparison",
            tags=["comparison", experiment_name]
        )
        
        print(f"\n📈 Creating comparison report: {experiment_name}")
        
        # Create comparison table
        columns = ["name", "consistency", "domain_accuracy", "inference_ms", "size_mb"]
        data = []
        
        for exp in experiments:
            data.append([
                exp.get('name', 'Unknown'),
                exp.get('consistency', 0.0),
                exp.get('domain_accuracy', 0.0),
                exp.get('inference_ms', 0.0),
                exp.get('size_mb', 0.0)
            ])
        
        comparison_table = wandb.Table(data=data, columns=columns)
        wandb.log({"experiment_comparison": comparison_table})
        
        # Create plots
        for metric in ['consistency', 'domain_accuracy', 'inference_ms']:
            if all(metric in exp for exp in experiments):
                values = [exp[metric] for exp in experiments]
                names = [exp.get('name', f'Exp{i}') for i, exp in enumerate(experiments)]
                
                plot_table = wandb.Table(
                    data=[[n, v] for n, v in zip(names, values)],
                    columns=["experiment", metric]
                )
                
                wandb.log({
                    f"{metric}_comparison": wandb.plot.bar(
                        plot_table,
                        "experiment",
                        metric,
                        title=f"{metric.replace('_', ' ').title()} Comparison"
                    )
                })
        
        print(f"  ✓ Compared {len(experiments)} experiments")
        
        self.finish_run()


# ========================================
# USAGE EXAMPLES
# ========================================

def demo_complete_pipeline():
    """
    Demonstrate complete monitoring pipeline
    """
    
    print("=" * 70)
    print("CLARA MONITORING - COMPLETE PIPELINE DEMO")
    print("=" * 70)
    
    # Initialize monitor
    monitor = ClaraMonitor(
        project_name="clara-deng-research",
        # api_key will be loaded from WB environment variable
    )
    
    # ========================================
    # DEMO 1: Track Data Generation
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 1: Track Data Generation")
    print("=" * 70)
    
    # Simulate generated data
    warmth_examples = [
        {"neutral": "That is correct.", "low": "Correct.", "high": "That's exactly right! Great job!"},
        {"neutral": "I can help.", "low": "Sure.", "high": "I'd be delighted to help you!"},
        {"neutral": "Here's the answer.", "low": "Answer:", "high": "Here's the answer you're looking for!"}
    ] * 10  # Simulate 30 examples
    
    monitor.track_data_generation(
        dimension="warmth",
        n_examples=len(warmth_examples),
        api_cost=5.50,
        examples=warmth_examples,
        quality_metrics={
            "human_validation_score": 0.92
        }
    )
    
    # ========================================
    # DEMO 2: Setup Training Tracking
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 2: Setup Training Tracking")
    print("=" * 70)
    
    training_config = monitor.setup_training_tracking(
        model_name="tinyllama_warmth",
        dimension="warmth",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hyperparameters={
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4
        }
    )
    
    # Simulate training logs
    for step in range(0, 101, 10):
        monitor.log_training_metrics({
            "train_loss": 2.0 - (step / 100) * 1.5,
            "learning_rate": 5e-5 * (1 - step / 100)
        }, step=step)
        time.sleep(0.1)  # Simulate training time
    
    monitor.finish_run()
    
    # ========================================
    # DEMO 3: Evaluate Personality
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 3: Evaluate Personality")
    print("=" * 70)
    
    # Mock generate function
    def mock_generate(prompt):
        return f"I'd be happy to help you with that! {prompt}"
    
    test_prompts = [
        "Can you help me?",
        "I'm stuck on this problem.",
        "Thanks for your help!"
    ]
    
    personality_metrics = monitor.evaluate_personality_consistency(
        model_path="./models/tinyllama_warmth",
        dimension="warmth",
        test_prompts=test_prompts,
        expected_traits={"warmth": 0.8},
        generate_fn=mock_generate
    )
    
    # ========================================
    # DEMO 4: Track Merge Experiment
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 4: Track Merge Experiment")
    print("=" * 70)
    
    merge_run = monitor.track_merge_experiment(
        merge_name="clara_v1_balanced",
        config_path="./configs/2_personality_plus_domain.yml",
        personality_weights={
            "warmth": 0.8,
            "formality": 0.3,
            "encouragement": 0.9
        },
        domain_weights={
            "robotics": 0.8,
            "python": 0.6
        },
        merge_method="linear"
    )
    
    # Simulate merge completion
    time.sleep(2)
    
    monitor.log_merge_results(
        merge_name="clara_v1_balanced",
        output_path="./models/clara_merged",
        model_size_mb=650.5,
        merge_time_seconds=127.3,
        evaluation_metrics={
            "personality_consistency": 0.87,
            "domain_accuracy_robotics": 0.82,
            "domain_accuracy_python": 0.78,
            "inference_speed_ms": 42.3
        }
    )
    
    # ========================================
    # DEMO 5: Evaluate HDC Memory (Mocked)
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 5: Evaluate HDC Memory")
    print("=" * 70)
    
    # Mock memory system
    class MockMemorySystem:
        def __init__(self):
            self.hdc_dimensions = 10000
            self.memories = [{}] * 50
        
        def query_similar(self, query, top_k=3):
            # Mock results
            return [
                ("I helped with Python async", 0.75),
                ("User asked about programming", 0.68),
                ("Discussed event loops", 0.62)
            ]
    
    mock_memory = MockMemorySystem()
    
    test_queries = [
        {"query": "Help with Python async"},
        {"query": "Robot grasping problem"},
        {"query": "Event loop explanation"}
    ]
    
    ground_truth = [
        "Python async",
        "Robot",
        "Event loop"
    ]
    
    memory_metrics = monitor.evaluate_hdc_memory(
        memory_system=mock_memory,
        test_queries=test_queries,
        ground_truth=ground_truth
    )
    
    # ========================================
    # DEMO 6: Create Comparison Report
    # ========================================
    
    print("\n" + "=" * 70)
    print("DEMO 6: Create Comparison Report")
    print("=" * 70)
    
    experiments = [
        {
            "name": "Clara v1 (Linear)",
            "consistency": 0.87,
            "domain_accuracy": 0.80,
            "inference_ms": 42.3,
            "size_mb": 650.5
        },
        {
            "name": "Clara v2 (DARE)",
            "consistency": 0.89,
            "domain_accuracy": 0.83,
            "inference_ms": 45.1,
            "size_mb": 655.2
        },
        {
            "name": "Clara v3 (TIES)",
            "consistency": 0.91,
            "domain_accuracy": 0.85,
            "inference_ms": 43.8,
            "size_mb": 652.8
        }
    ]
    
    monitor.create_comparison_report(
        experiment_name="merge_methods",
        experiments=experiments
    )
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 70)
    print("MONITORING COMPLETE!")
    print("=" * 70)
    
    print(f"""
✓ Data generation tracked
✓ Fine-tuning monitored
✓ Personality evaluated
✓ Merge experiment logged
✓ HDC memory benchmarked
✓ Comparison report created

🎯 View your results at:
   https://wandb.ai/your-username/clara-deng-research

📊 Available dashboards:
   - Training curves
   - Personality consistency
   - Merge comparisons
   - Memory performance
   - Experiment comparisons
    """)


if __name__ == "__main__":
    demo_complete_pipeline()