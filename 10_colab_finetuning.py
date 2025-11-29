"""
Google Colab Pro+ Fine-tuning Setup
Optimized for A100/V100 GPUs
"""

def create_colab_notebook_code():
    """
    Generate code for Colab notebook
    """
    
    print("=" * 70)
    print("COLAB PRO+ FINE-TUNING SETUP")
    print("=" * 70)
    
    notebook_code = '''
# ============================================================
# CLARA FINE-TUNING - COLAB PRO+ NOTEBOOK
# ============================================================

# CELL 1: Setup and Installation
# ============================================================
!pip install -q transformers datasets accelerate wandb bitsandbytes
!pip install -q peft  # For LoRA fine-tuning (optional but recommended)

import wandb
wandb.login()  # Paste your API key

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
!mkdir -p /content/drive/MyDrive/clara_models
!mkdir -p /content/drive/MyDrive/clara_data

# CELL 2: Check GPU
# ============================================================
!nvidia-smi

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# CELL 3: Load Data from Drive
# ============================================================
import json
from pathlib import Path

# Load your personality data
data_path = Path("/content/drive/MyDrive/clara_data")

with open(data_path / "warmth_training.json") as f:
    warmth_data = json.load(f)

print(f"Loaded {len(warmth_data)} training examples")
print(f"Example: {warmth_data[0]}")

# CELL 4: Prepare Dataset for HuggingFace
# ============================================================
from datasets import Dataset

def format_for_training(examples):
    """Convert personality data to instruction format"""
    formatted = []
    
    for ex in examples:
        # Create training pair: neutral → transformed
        formatted.append({
            "instruction": f"Rewrite this with high warmth: {ex['neutral']}",
            "output": ex['high']
        })
        formatted.append({
            "instruction": f"Rewrite this with low warmth: {ex['neutral']}",
            "output": ex['low']
        })
    
    return formatted

formatted_data = format_for_training(warmth_data)
dataset = Dataset.from_list(formatted_data)

# Split into train/val
dataset = dataset.train_test_split(test_size=0.1)

print(f"Train examples: {len(dataset['train'])}")
print(f"Val examples: {len(dataset['test'])}")

# CELL 5: Load Base Model (with 4-bit quantization for memory efficiency)
# ============================================================
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# CELL 6: Configure LoRA (Parameter-Efficient Fine-Tuning)
# ============================================================
# LoRA lets us fine-tune with <1% of parameters!
# Perfect for Colab

lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
print(f"Total parameters: {model.num_parameters():,}")
print(f"Trainable %: {100 * model.num_parameters(only_trainable=True) / model.num_parameters():.2f}%")

# CELL 7: Tokenize Dataset
# ============================================================
def tokenize_function(examples):
    # Format as instruction-following
    prompts = [
        f"### Instruction: {inst}\\n### Response: {out}"
        for inst, out in zip(examples['instruction'], examples['output'])
    ]
    
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)

# CELL 8: Training Configuration with W&B Logging
# ============================================================
output_dir = "/content/drive/MyDrive/clara_models/tinyllama_warmth"

training_args = TrainingArguments(
    output_dir=output_dir,
    
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Optimization
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    fp16=True,
    
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    evaluation_strategy="steps",
    save_total_limit=3,
    
    # W&B integration
    report_to="wandb",
    run_name="clara-warmth-finetune",
    
    # Memory optimization
    gradient_checkpointing=True,
    max_grad_norm=0.3
)

# CELL 9: Create Trainer and Start Training
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

# Start training!
print("🚀 Starting training...")
print("View progress at: https://wandb.ai")

trainer.train()

# CELL 10: Save Model
# ============================================================
# Save LoRA adapters (very small!)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to: {output_dir}")
print(f"  Size: ~{sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file()) / 1e6:.1f} MB")

# CELL 11: Test the Fine-tuned Model
# ============================================================
from peft import PeftModel

# Load for inference
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto"
)
model_reload = PeftModel.from_pretrained(base_model_reload, output_dir)

def generate_warm_response(prompt, warmth_level="high"):
    instruction = f"Rewrite this with {warmth_level} warmth: {prompt}"
    inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
    
    outputs = model_reload.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
test_prompt = "I can help with that."
print(f"Original: {test_prompt}")
print(f"High warmth: {generate_warm_response(test_prompt, 'high')}")
print(f"Low warmth: {generate_warm_response(test_prompt, 'low')}")

# CELL 12: Download to Drive (Already Saved!)
# ============================================================
print("Model is automatically saved to Google Drive!")
print(f"Location: {output_dir}")
print("You can download this to your local machine or use for merging")
'''
    
    print("\n📓 Colab Notebook Code Generated!")
    print("\nTo use:")
    print("  1. Create new Colab notebook")
    print("  2. Copy cells above")
    print("  3. Run with A100 GPU (Colab Pro+)")
    
    print("\n⚡ Optimization for Colab Pro+:")
    print("  - Uses 4-bit quantization (saves memory)")
    print("  - LoRA fine-tuning (<1% parameters)")
    print("  - Gradient checkpointing")
    print("  - Fits on T4/V100/A100")
    
    print("\n💾 Storage:")
    print("  - Models save to Google Drive automatically")
    print("  - LoRA adapters are tiny (~50MB)")
    print("  - Can fine-tune all 7 models in one day!")
    
    # Save to file
    with open("clara_colab_finetune.py", "w") as f:
        f.write(notebook_code)
    
    print("\n✓ Saved to: clara_colab_finetune.py")
    print("  Copy this into Colab cells")


if __name__ == "__main__":
    create_colab_notebook_code()