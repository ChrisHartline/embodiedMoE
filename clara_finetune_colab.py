# ============================================================
# CLARA FINE-TUNING NOTEBOOK - Google Colab Pro+
# ============================================================
# 
# Instructions:
# 1. Upload this to Google Colab
# 2. Set Runtime > Change runtime type > A100 GPU
# 3. Run all cells
# 4. Model saves to Google Drive automatically
#
# ============================================================

# CELL 1: Setup and Installation
# ============================================================
!pip install -q transformers datasets accelerate wandb bitsandbytes
!pip install -q peft trl
!pip install -q sentencepiece

import wandb
wandb.login()  # Paste your WB API key when prompted

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("✓ Setup complete!")

# CELL 2: Configuration - EDIT THIS
# ============================================================
# Which dimension/domain are you training?
DIMENSION = "warmth"  # Change this: warmth, playful, formal, encouragement, medical, coding, teaching, quantum

# Paths
DATA_PATH = f"/content/drive/MyDrive/Lily/training_data/{DIMENSION}_training.json"
OUTPUT_DIR = f"/content/drive/MyDrive/Lily/models/tinyllama_{DIMENSION}"

# Base model
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Training config
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

print(f"Training: {DIMENSION}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

# CELL 3: Check GPU
# ============================================================
!nvidia-smi

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# CELL 4: Load and Prepare Data
# ============================================================
import json
from datasets import Dataset

# Load your training data
with open(DATA_PATH) as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} examples")
print(f"Sample: {raw_data[0]}")

# Format for training
def format_personality_data(examples):
    """Format personality data for instruction fine-tuning"""
    formatted = []
    
    for ex in examples:
        if 'neutral' in ex:  # Personality data format
            # Train to transform neutral → high
            formatted.append({
                "instruction": f"Rewrite this with high {DIMENSION}: {ex['neutral']}",
                "response": ex['high']
            })
            # Also train neutral → low for contrast
            formatted.append({
                "instruction": f"Rewrite this with low {DIMENSION}: {ex['neutral']}",
                "response": ex['low']
            })
        elif 'question' in ex:  # Domain knowledge format
            formatted.append({
                "instruction": ex['question'],
                "response": ex['answer']
            })
    
    return formatted

formatted_data = format_personality_data(raw_data)
print(f"Formatted into {len(formatted_data)} training examples")

# Create dataset
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1)

print(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")

# CELL 5: Load Model with Quantization
# ============================================================
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✓ Model loaded!")

# CELL 6: Configure LoRA
# ============================================================
# LoRA = Low-Rank Adaptation (trains <1% of parameters)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# CELL 7: Prepare Training Data
# ============================================================
def create_prompt(instruction, response=""):
    """Create training prompt in chat format"""
    if response:
        return f"""### Instruction:
{instruction}

### Response:
{response}"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
"""

def tokenize_function(examples):
    """Tokenize examples"""
    prompts = [
        create_prompt(inst, resp) 
        for inst, resp in zip(examples['instruction'], examples['response'])
    ]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)
print("✓ Tokenization complete!")

# CELL 8: Training Configuration
# ============================================================
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    
    # Optimizer
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    
    # Precision
    fp16=True,
    
    # Logging
    logging_steps=10,
    logging_first_step=True,
    
    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=50,
    
    # W&B
    report_to="wandb",
    run_name=f"clara-{DIMENSION}",
    
    # Memory
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

print("✓ Training config ready!")

# CELL 9: Train!
# ============================================================
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
)

print("🚀 Starting training...")
print(f"   Watch progress: https://wandb.ai/chris_hartline/clara-deng-research")

trainer.train()

print("✓ Training complete!")

# CELL 10: Save Model
# ============================================================
print(f"Saving to {OUTPUT_DIR}...")

# Save the LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✓ Model saved!")

# Calculate size
import os
total_size = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
    for f in os.listdir(OUTPUT_DIR) 
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
)
print(f"   Size: {total_size / 1e6:.1f} MB")

# CELL 11: Test the Model
# ============================================================
print("\nTesting trained model...")

def generate_response(prompt, max_length=100):
    full_prompt = create_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Test prompts
test_prompts = [
    f"Rewrite this with high {DIMENSION}: I can help you with that.",
    f"Rewrite this with high {DIMENSION}: That's correct.",
    f"Rewrite this with low {DIMENSION}: I can help you with that.",
]

print("\nTest Results:")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print(f"Response: {generate_response(prompt)}")

# CELL 12: Done!
# ============================================================
print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"""
Model saved to: {OUTPUT_DIR}

Next steps:
1. Run this notebook again with different DIMENSION
2. After all dimensions trained, run mergekit
3. Check W&B for training metrics

Dimensions to train:
- warmth
- playful  
- formal
- encouragement
- medical
- coding
- teaching
- quantum
""")

wandb.finish()