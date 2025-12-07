# CLAUDE.md - Embodied MoE Project Guide

## Project Overview

**embodiedMoE** is a research project for creating "Clara", an embodied AI assistant with configurable personality traits. The project combines:

- **Hyperdimensional Computing (HDC)** for efficient associative memory
- **Mixture of Experts (MoE)** through model merging to blend personality dimensions
- **TinyLlama** (1.1B parameters) as the lightweight base model for embodied/robotics applications
- **LoRA fine-tuning** on personality and domain-specific datasets
- **Mergekit** for combining fine-tuned models into a single model with blended traits

## Project Structure

```
embodiedMoE/
├── clara_master.py           # Main pipeline orchestrator
├── clara_complete_pipeline.py # End-to-end pipeline with W&B integration
├── clara_finetune.py         # Google Colab fine-tuning notebook (jupytext)
├── clara_monitoring_complete.py # W&B monitoring system
├── clara_spec.json           # Clara's personality/expertise specification
├── config.py                 # Global configuration settings
├── 1_hdc_basics.py           # HDC memory fundamentals (bind/bundle/query)
├── 2_personality_data.py     # Claude API data generation for personality
├── 3_tiny_model.py           # TinyLlama model loading demo
├── 4_clara_prototype.py      # Initial Clara prototype with HDC
├── 5_clara_with_embeddings.py # Clara with neural embeddings
├── 6_mergekit_personality.py # Mergekit configuration creation
├── 7_create_clara_full_pipeline.py # Full pipeline demonstration
├── 8_phi_merge_example.py    # Phi model merge example
├── 9_monitoring_setup.py     # W&B monitoring setup
├── 10_colab_finetuning.py    # Colab training guide
├── 12_clara_merge_configs.py # Advanced merge configurations
├── hdc_with_embeddings.py    # HDC memory with sentence embeddings
├── regenerate_missing_data.py # Utility to regenerate training data
├── test_setup.py             # Environment verification script
├── configs/
│   ├── clara_full_merge.yaml # Mergekit configuration
│   └── clara_merge.json      # Merge weights JSON
├── data/
│   ├── warmth_training.json      # Personality data
│   ├── playful_training.json
│   ├── formal_training.json
│   ├── encouragement_training.json
│   ├── medical_knowledge.json    # Domain data
│   ├── coding_knowledge.json
│   ├── teaching_knowledge.json
│   └── quantum_knowledge.json
├── models/                   # Fine-tuned models (gitignored)
├── results/                  # Training outputs and instructions
└── wandb/                    # W&B experiment logs
```

## Key Concepts

### 1. Hyperdimensional Computing (HDC)

HDC uses high-dimensional binary vectors (10,000 dimensions) for memory:
- **Bind**: Element-wise multiplication to create associations
- **Bundle**: Superposition (sum and sign) to combine concepts
- **Query**: Similarity-based retrieval using dot products

Example from `1_hdc_basics.py`:
```python
memory = HDCMemory(dimensions=10000)
memory.store_memory(subject="CLARA", action="PICKED_UP", object="CUP", result="SUCCESS")
results = memory.query(object="CUP", result="SUCCESS")  # Returns action
```

### 2. Personality Dimensions

Clara has 4 personality dimensions (0.0-1.0 scale):
- **warmth**: Friendly/caring (0.8) vs cold/distant
- **playful**: Fun/witty (0.7) vs serious/straightforward
- **formal**: Professional (0.3) vs casual
- **encouragement**: Supportive (0.9) vs neutral

### 3. Domain Expertise

Clara's knowledge domains:
- **medical**: Clinical reasoning, terminology (0.7)
- **coding**: Python programming, async/await (0.8)
- **teaching**: Explaining concepts, analogies (0.9)
- **quantum**: Qiskit, quantum circuits (0.8)

### 4. Training Data Format

**Personality data** (warmth_training.json, etc.):
```json
{
  "neutral": "I can help with that.",
  "low": "Sure.",
  "high": "I'd be absolutely delighted to help you with that!",
  "dimension": "warmth"
}
```

**Domain data** (medical_knowledge.json, etc.):
```json
{
  "question": "What is tachycardia?",
  "answer": "Tachycardia is a heart rate exceeding 100 beats per minute...",
  "difficulty": "beginner",
  "domain": "medical"
}
```

## Development Workflow

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (separate)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "WANDB_API_KEY=your_key_here" >> .env

# Verify setup
python test_setup.py
```

### 2. Data Generation

Generate personality/domain training data using Claude API:

```bash
# Run master pipeline (generates all data)
python clara_master.py

# Or generate specific dimension
python 2_personality_data.py
```

Data generation uses Claude Sonnet to create high/neutral/low examples for each personality dimension.

### 3. Fine-tuning (Google Colab)

Fine-tuning requires GPU and is done in Google Colab:

1. Upload data files to Google Drive
2. Open `clara_finetune.py` in Colab
3. Set `DIMENSION = "warmth"` (or other dimension)
4. Run all cells
5. Repeat for each dimension/domain

Training configuration:
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Method: LoRA (r=16, alpha=32)
- Epochs: 3
- Batch size: 4 (effective 16 with grad accumulation)
- Precision: 4-bit quantization

### 4. Model Merging

After fine-tuning all dimensions, merge with mergekit:

```bash
# Create merge config
python 6_mergekit_personality.py

# Run merge
mergekit-yaml ./configs/clara_full_merge.yaml ./models/clara_merged --cuda
```

Merge methods supported:
- `linear`: Simple weighted average
- `ties`: Task-specific merging
- `dare`: Density-Aware merging
- `slerp`: Spherical linear interpolation

### 5. Monitoring with W&B

All experiments are tracked in Weights & Biases:

```python
from clara_monitoring_complete import ClaraMonitor
monitor = ClaraMonitor(project_name="clara-deng-research")
monitor.track_data_generation(dimension="warmth", n_examples=1000, ...)
```

View dashboard at: `https://wandb.ai/chris_hartline/clara-deng-research`

## Important Files

### clara_master.py (Main Entry Point)

Orchestrates the complete pipeline:
- Loads `clara_spec.json` for personality/expertise configuration
- Generates training data via Claude API
- Prepares fine-tuning instructions
- Tracks progress in W&B

### clara_spec.json (Configuration)

```json
{
  "personality": {"warmth": 0.8, "playful": 0.7, "formal": 0.3, "encouragement": 0.9},
  "expertise": {"medical": 0.7, "coding": 0.8, "teaching": 0.9, "quantum": 0.8},
  "config": {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "examples_per_dimension": 1000,
    "merge_method": "ties"
  }
}
```

### config.py (Global Settings)

```python
HDC_DIMENSIONS = 10000
TINY_MODEL_NAME = "roneneldan/TinyStories-33M"
DEFAULT_PERSONALITY = {'warmth': 0.8, 'formality': 0.3, 'verbosity': 0.6, 'encouragement': 0.9}
DATA_DIR = "./data"
MODELS_DIR = "./models"
```

## Common Tasks

### Run the Master Pipeline
```bash
python clara_master.py
```

### Test HDC Memory
```bash
python 1_hdc_basics.py
```

### Test HDC with Neural Embeddings
```bash
python hdc_with_embeddings.py
```

### Regenerate Missing Data
```bash
python regenerate_missing_data.py
```

### Check CUDA Availability
```bash
python check_cuda.py
```

## Environment Variables

Required in `.env`:
- `ANTHROPIC_API_KEY`: For data generation via Claude API
- `WANDB_API_KEY` or `WB`: For experiment tracking

## Dependencies

Core dependencies (`requirements.txt`):
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `transformers>=4.40.0`
- `datasets>=2.14.0`
- `accelerate>=0.27.0`
- `anthropic>=0.28.0`
- `mergekit`
- `wandb`
- PyTorch (install separately with CUDA support)

## Code Conventions

1. **Numbered files** (1_*.py through 12_*.py): Tutorial/demo scripts in learning order
2. **clara_*.py files**: Production pipeline components
3. **Data format**: JSON with dimension/domain tags
4. **Model paths**: `./models/tinyllama_<dimension>/`
5. **Config paths**: `./configs/` for merge configurations

## Testing

```bash
# Verify environment
python test_setup.py

# Run HDC demo
python 1_hdc_basics.py

# Check data files
python check_data.py
```

## Notes for AI Assistants

1. **API Keys**: Never commit `.env` files. Check for `ANTHROPIC_API_KEY` before data generation.

2. **GPU Requirements**: Fine-tuning requires GPU. Use Google Colab Pro+ with A100 for best performance.

3. **Model Sizes**: TinyLlama is 1.1B parameters. Merged models are ~2-4GB on disk.

4. **Data Generation Cost**: Generating 1000 examples per dimension costs approximately $5-10 in Claude API calls.

5. **Merge Weights**: Personality weights in `clara_spec.json` control the blend. Higher weight = stronger trait expression.

6. **HDC Dimensions**: 10,000 dimensions is the default. This provides ~99% accuracy for associative memory with thousands of memories.

7. **W&B Entity**: The default W&B entity is `chris_hartline`. Update in `clara_monitoring_complete.py` if using a different account.

8. **File Encoding**: Training data JSON files should be UTF-8 encoded.

9. **Empty Data Files**: Some domain data files (coding, quantum, teaching) may be empty (`[]`) if not yet generated.

10. **Colab Runtime**: Use A100 GPU for ~30 min training per dimension, V100 for ~60 min, T4 for ~90 min.
