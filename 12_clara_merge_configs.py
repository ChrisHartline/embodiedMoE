"""
Clara Merge Configurations
Different strategies for creating Clara's personality + knowledge
"""

def config_1_simple_personality():
    """
    Simple: Merge personality-tuned models
    """
    
    config = """
# ============================================================
# CONFIG 1: Simple Personality Merge
# ============================================================
# Merges 3 personality-tuned TinyLlama models

%%writefile clara_personality_v1.yml
slices:
  - sources:
      - model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        layer_range: [0, 22]
      - model: './models/tinyllama_warm'
        layer_range: [0, 22]
      - model: './models/tinyllama_encouraging'
        layer_range: [0, 22]
merge_method: linear
weight: [0.2, 0.5, 0.3]  # 20% base, 50% warm, 30% encouraging
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_personality_v1
tokenizer_source: base
"""
    
    return config


def config_2_personality_plus_domain():
    """
    Advanced: Personality + Domain Knowledge
    """
    
    config = """
# ============================================================
# CONFIG 2: Personality + Domain Knowledge
# ============================================================
# Combines personality traits with robotics expertise

%%writefile clara_full_v1.yml
slices:
  - sources:
      # Base model (foundation)
      - model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        layer_range: [0, 22]
      
      # Personality models
      - model: './models/tinyllama_warm'
        layer_range: [0, 22]
      - model: './models/tinyllama_casual'
        layer_range: [0, 22]
      - model: './models/tinyllama_encouraging'
        layer_range: [0, 22]
      
      # Domain knowledge models
      - model: './models/tinyllama_robotics'
        layer_range: [0, 22]
      - model: './models/tinyllama_python'
        layer_range: [0, 22]

merge_method: linear
weight: [0.15, 0.25, 0.15, 0.15, 0.20, 0.10]
# Base 15%, Warm 25%, Casual 15%, Encouraging 15%, Robotics 20%, Python 10%

base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_full_v1
tokenizer_source: base

parameters:
  # Ensure embedding layers are properly merged
  - name: model.embed_tokens.weight
    method: linear
    weight: [0.15, 0.25, 0.15, 0.15, 0.20, 0.10]
  
  - name: lm_head.weight
    method: linear
    weight: [0.15, 0.25, 0.15, 0.15, 0.20, 0.10]
"""
    
    return config


def config_3_cross_architecture():
    """
    Experimental: Merge different architectures
    """
    
    config = """
# ============================================================
# CONFIG 3: Cross-Architecture Merge (Experimental)
# ============================================================
# Mix TinyLlama with Phi for better capabilities

%%writefile clara_hybrid.yml
slices:
  - sources:
      # TinyLlama (warm personality)
      - model: './models/tinyllama_warm'
        layer_range: [0, 22]
      
      # Phi-2 (better reasoning, 2.7B params)
      # Using fewer layers to keep size reasonable
      - model: 'microsoft/phi-2'
        layer_range: [0, 16]  # Only first 16 layers

merge_method: slerp  # Spherical interpolation (better for different archs)
t: 0.7  # 70% TinyLlama, 30% Phi-2
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_hybrid

# Note: This is experimental! 
# Different architectures may not merge cleanly
# But when it works, you get best of both worlds
"""
    
    return config


def config_4_layer_specific():
    """
    Advanced: Different merges for different layers
    """
    
    config = """
# ============================================================
# CONFIG 4: Layer-Specific Merge
# ============================================================
# Early layers: general knowledge
# Middle layers: personality
# Late layers: domain expertise

%%writefile clara_layered.yml
slices:
  # Early layers (0-7): Keep base model strong
  - sources:
      - model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        layer_range: [0, 7]
      - model: './models/tinyllama_robotics'
        layer_range: [0, 7]
    merge_method: linear
    weight: [0.8, 0.2]  # Mostly base for foundation
  
  # Middle layers (8-15): Focus on personality
  - sources:
      - model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        layer_range: [8, 15]
      - model: './models/tinyllama_warm'
        layer_range: [8, 15]
      - model: './models/tinyllama_encouraging'
        layer_range: [8, 15]
    merge_method: linear
    weight: [0.2, 0.5, 0.3]  # Heavy personality
  
  # Late layers (16-22): Domain specialization
  - sources:
      - model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        layer_range: [16, 22]
      - model: './models/tinyllama_robotics'
        layer_range: [16, 22]
      - model: './models/tinyllama_python'
        layer_range: [16, 22]
    merge_method: linear
    weight: [0.3, 0.5, 0.2]  # Heavy domain knowledge

base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_layered
"""
    
    return config


def config_5_dare_method():
    """
    DARE method: Drop And REscale (better for many models)
    """
    
    config = """
# ============================================================
# CONFIG 5: DARE Method (Best for 5+ models)
# ============================================================
# DARE randomly drops parameters to avoid interference

%%writefile clara_dare.yml
merge_method: dare_linear
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_dare

models:
  - model: './models/tinyllama_warm'
    parameters:
      density: 0.7  # Keep 70% of delta (difference from base)
      weight: 0.3
  
  - model: './models/tinyllama_casual'
    parameters:
      density: 0.6
      weight: 0.2
  
  - model: './models/tinyllama_encouraging'
    parameters:
      density: 0.7
      weight: 0.3
  
  - model: './models/tinyllama_robotics'
    parameters:
      density: 0.8  # Keep more domain knowledge
      weight: 0.4
  
  - model: './models/tinyllama_python'
    parameters:
      density: 0.7
      weight: 0.2

# DARE is good when merging many models
# Reduces parameter interference
# Often gives better results than simple linear
"""
    
    return config


def config_6_ties_method():
    """
    TIES method: Trim, Elect, Merge (state-of-the-art)
    """
    
    config = """
# ============================================================
# CONFIG 6: TIES Method (State-of-the-art)
# ============================================================
# TIES: Resolves parameter conflicts intelligently

%%writefile clara_ties.yml
merge_method: ties
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dtype: q4_k_m
out_path: ./clara_ties

parameters:
  density: 0.5  # Trim 50% of smallest deltas
  weight:
    - model: './models/tinyllama_warm'
      value: 0.3
    - model: './models/tinyllama_casual'
      value: 0.2
    - model: './models/tinyllama_encouraging'
      value: 0.3
    - model: './models/tinyllama_robotics'
      value: 0.4
    - model: './models/tinyllama_python'
      value: 0.2

# TIES Process:
# 1. TRIM: Remove small, likely noisy changes
# 2. ELECT: Resolve conflicting changes by voting
# 3. MERGE: Combine elected changes
# Often best quality, especially with many models!
"""
    
    return config


def create_all_configs():
    """
    Generate all Clara merge configurations
    """
    
    print("=" * 70)
    print("CLARA MERGE CONFIGURATIONS")
    print("=" * 70)
    
    configs = {
        "1_simple_personality": config_1_simple_personality(),
        "2_personality_plus_domain": config_2_personality_plus_domain(),
        "3_cross_architecture": config_3_cross_architecture(),
        "4_layer_specific": config_4_layer_specific(),
        "5_dare_method": config_5_dare_method(),
        "6_ties_method": config_6_ties_method()
    }
    
    print("\n📋 Generated 6 merge configurations:")
    
    for name, config in configs.items():
        filename = f"./configs/{name}.yml"
        
        # Extract just the YAML part
        yaml_start = config.find("%%writefile")
        if yaml_start > 0:
            yaml_content = config[yaml_start:].split('\n', 1)[1]
        else:
            yaml_content = config
        
        print(f"\n  {name}")
        print(f"    File: {filename}")
        
        # Save config
        import os
        os.makedirs("./configs", exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(yaml_content)
        
        print(f"    ✓ Saved")
    
    print("\n" + "=" * 70)
    print("WHICH CONFIG TO USE?")
    print("=" * 70)
    
    recommendations = """
┌─────────────────────┬────────────────────────────────────────┐
│ Configuration       │ When to Use                            │
├─────────────────────┼────────────────────────────────────────┤
│ 1. Simple           │ • First experiments                    │
│ Personality         │ • Testing merge concept                │
│                     │ • Quick iterations                     │
├─────────────────────┼────────────────────────────────────────┤
│ 2. Personality +    │ • Full Clara (recommended!)            │
│ Domain              │ • Production deployment                │
│                     │ • D.Eng main results                   │
├─────────────────────┼────────────────────────────────────────┤
│ 3. Cross-           │ • Experimental chapter                 │
│ Architecture        │ • If you want better reasoning         │
│                     │ • Willing to accept larger model       │
├─────────────────────┼────────────────────────────────────────┤
│ 4. Layer-Specific   │ • Advanced optimization                │
│                     │ • After simpler merges work            │
│                     │ • Squeezing out extra performance      │
├─────────────────────┼────────────────────────────────────────┤
│ 5. DARE             │ • Merging 5+ models                    │
│                     │ • Parameter interference issues        │
│                     │ • Need stability                       │
├─────────────────────┼────────────────────────────────────────┤
│ 6. TIES             │ • Best quality (state-of-the-art)      │
│                     │ • Final production merge               │
│                     │ • Publication results                  │
└─────────────────────┴────────────────────────────────────────┘
    """
    
    print(recommendations)
    
    print("\n💡 Recommended Progression:")
    print("  Week 1-2: Config 1 (Simple) - Learn the process")
    print("  Week 3-4: Config 2 (Full) - Main research version")
    print("  Week 5-6: Config 6 (TIES) - Optimize for best results")
    print("  Optional: Config 3 (Hybrid) - Experimental comparison")


def usage_example():
    """
    Show how to actually run mergekit
    """
    
    print("\n" + "=" * 70)
    print("HOW TO RUN MERGEKIT")
    print("=" * 70)
    
    commands = """
# Install mergekit
pip install -U mergekit

# Method 1: Using config file
mergekit-yaml ./configs/2_personality_plus_domain.yml ./models/clara_merged

# Method 2: With specific options
mergekit-yaml \\
  ./configs/2_personality_plus_domain.yml \\
  ./models/clara_merged \\
  --copy-tokenizer \\
  --allow-crimes \\  # Allow experimental merges
  --cuda  # Use GPU for faster merging

# Method 3: In Colab (for large merges)
!pip install -q mergekit
!mergekit-yaml ./configs/2_personality_plus_domain.yml /content/drive/MyDrive/clara_merged --cuda

# Monitor progress
# Mergekit shows:
# - Current layer being merged
# - Memory usage
# - Estimated time remaining

# Typical merge times:
# - TinyLlama models: 5-10 minutes (CPU)
# - TinyLlama models: 1-2 minutes (GPU)
# - Phi-2 models: 15-30 minutes (GPU)
# - Cross-architecture: 20-40 minutes (GPU)
    """
    
    print(commands)
    
    print("\n✅ After Merging:")
    print("""
Your merged model will be in: ./models/clara_merged/
  ├── config.json
  ├── model-00001-of-00002.safetensors
  ├── model-00002-of-00002.safetensors
  ├── tokenizer.json
  └── tokenizer_config.json

Test it:
  
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/clara_merged")
tokenizer = AutoTokenizer.from_pretrained("./models/clara_merged")

inputs = tokenizer("Hello Clara!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
    """)


def compare_to_example():
    """
    Compare Clara approach to the provided example
    """
    
    print("\n" + "=" * 70)
    print("CLARA vs PROVIDED EXAMPLE")
    print("=" * 70)
    
    comparison = """
PROVIDED EXAMPLE:
─────────────────────────────────────────────────────────────
Merges: Phi-3.5-mini (3.8B) + Hermes-Mistral-7B (7B)
Goal: Combine instruction-following + reasoning
Weights: 65% Phi, 35% Mistral
Result: ~3.8B param model with both capabilities
Output: q4_k_m quantized (~2.5GB)

CLARA APPROACH (Config 2):
─────────────────────────────────────────────────────────────
Merges: Base + 3 personality + 2 domain (6 models total)
Goal: Personality + Domain expertise from fine-tunes
Weights: Distributed across personality & knowledge
Result: ~1.1B param model with Clara's traits
Output: q4_k_m quantized (~600MB)

KEY DIFFERENCES:
─────────────────────────────────────────────────────────────
Example:
  ✓ Merges different base models
  ✓ Combines existing capabilities
  ✓ Larger, more capable
  ✗ No explicit personality control
  ✗ Can't fine-tune individual aspects

Clara:
  ✓ Merges fine-tuned variants
  ✓ Explicit personality dimensions
  ✓ Much smaller (edge-deployable)
  ✓ Can adjust weights per dimension
  ✗ Limited by base model capabilities

BOTH APPROACHES ARE VALID!
─────────────────────────────────────────────────────────────
Example approach: When you need raw capability
Clara approach: When you need specific personality + efficiency
    """
    
    print(comparison)


if __name__ == "__main__":
    create_all_configs()
    usage_example()
    compare_to_example()