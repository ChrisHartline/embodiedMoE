# %% [markdown]
# # CLARA FINE-TUNING NOTEBOOK - Google Colab Pro+
# 
# Instructions:
# 1. Upload this to Google Colab
# 2. Set Runtime > Change runtime type > A100 GPU
# 3. Run all cells
# 4. Model saves to Google Drive automatically

# %% [markdown]
# ## Cell 1: Setup and Installation

# %%
!pip install -q transformers datasets accelerate wandb bitsandbytes
!pip install -q peft trl
!pip install -q sentencepiece

import wandb
wandb.login()  # Paste your WB API key when prompted

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("✓ Setup complete!")

# %% [markdown]
# ## Cell 2: Configuration - EDIT THIS!

# %%
# ============================================================
# WHICH DIMENSION/DOMAIN ARE YOU TRAINING?
# ============================================================
# Change this for each training run:
#   Personality: warmth, playful, formal, encouragement
#   Domain: medical, coding, teaching, quantum

DIMENSION = "warmth"  # <-- CHANGE THIS EACH RUN

# ============================================================
# PATHS - Adjust if your data is in a different location
# ============================================================
# For personality data (warmth, playful, formal, encouragement):
if DIMENSION in ["warmth", "playful", "formal", "encouragement"]:
    DATA_PATH = f"/content/drive/MyDrive/Lily/training_data/{DIMENSION}_training.json"
# For domain data (medical, coding, teaching, quantum):
else:
    DATA_PATH = f"/content/drive/MyDrive/Lily/training_data/{DIMENSION}_knowledge.json"

OUTPUT_DIR = f"/content/drive/MyDrive/Lily/models/tinyllama_{DIMENSION}"

# ============================================================
# BASE MODEL
# ============================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ============================================================
# TRAINING CONFIG
# ============================================================
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

print(f"=" * 60)
print(f"TRAINING: {DIMENSION}")
print(f"=" * 60)
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Base model: {BASE_MODEL}")
print(f"Epochs: {EPOCHS}")

# %% [markdown]
# ## Cell 3: Check GPU

# %%
!nvidia-smi

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")

# %% [markdown]
# ## Cell 4: Load and Prepare Data

# %%
import json
from datasets import Dataset
import os

# Check if file exists
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Data file not found: {DATA_PATH}")
    print("\nAvailable files in training_data:")
    training_dir = "/content/drive/MyDrive/Lily/training_data"
    if os.path.exists(training_dir):
        for f in os.listdir(training_dir):
            print(f"  - {f}")
    else:
        print(f"  Directory not found: {training_dir}")
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

# Load your training data
with open(DATA_PATH) as f:
    raw_data = json.load(f)

print(f"✓ Loaded {len(raw_data)} examples")
print(f"\nSample example:")
print(json.dumps(raw_data[0], indent=2))

# Format for training
def format_training_data(examples, dimension):
    """Format data for instruction fine-tuning"""
    formatted = []
    
    for ex in examples:
        if 'neutral' in ex:  # Personality data format
            # Train to transform neutral → high
            formatted.append({
                "instruction": f"Rewrite this with high {dimension}: {ex['neutral']}",
                "response": ex['high']
            })
            # Also train neutral → low for contrast
            formatted.append({
                "instruction": f"Rewrite this with low {dimension}: {ex['neutral']}",
                "response": ex['low']
            })
        elif 'question' in ex:  # Domain knowledge format
            formatted.append({
                "instruction": ex['question'],
                "response": ex['answer']
            })
        else:
            print(f"Warning: Unknown format: {ex.keys()}")
    
    return formatted

formatted_data = format_training_data(raw_data, DIMENSION)
print(f"\n✓ Formatted into {len(formatted_data)} training examples")

# Create dataset
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"  Train: {len(dataset['train'])}")
print(f"  Val: {len(dataset['test'])}")

# %% [markdown]
# ## Cell 5: Load Model with Quantization

# %%
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

print("Loading model (this takes 1-2 minutes)...")
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

# %% [markdown]
# ## Cell 6: Configure LoRA

# %%
# LoRA = Low-Rank Adaptation
# Trains only ~1% of parameters, much faster and uses less memory

lora_config = LoraConfig(
    r=16,                # Rank of update matrices
    lora_alpha=32,       # Scaling factor
    target_modules=[     # Which layers to adapt
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✓ LoRA configured!")
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# %% [markdown]
# ## Cell 7: Prepare Training Data

# %%
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
    """Tokenize examples for training"""
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
    
    # Labels are same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)
print("✓ Tokenization complete!")

# %% [markdown]
# ## Cell 8: Training Configuration

# %%
import os

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    
    # Optimizer
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    
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
    
    # W&B Integration
    report_to="wandb",
    run_name=f"clara-{DIMENSION}",
    
    # Memory optimization
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

print("✓ Training configuration ready!")
print(f"\n  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * 4})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  W&B run: clara-{DIMENSION}")

# %% [markdown]
# ## Cell 9: Train!

# %%
from transformers import DataCollatorForLanguageModeling, Trainer

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
)

print("=" * 60)
print(f"🚀 STARTING TRAINING: {DIMENSION}")
print("=" * 60)
print(f"\nWatch progress at:")
print(f"  https://wandb.ai/chris_hartline/clara-deng-research")
print(f"\nEstimated time: 30-60 minutes on A100")
print("=" * 60)

# Train!
trainer.train()

print("\n✓ Training complete!")

# %% [markdown]
# ## Cell 10: Save Model

# %%
print(f"\nSaving model to: {OUTPUT_DIR}")

# Save the LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Calculate size
total_size = 0
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))

print(f"✓ Model saved!")
print(f"  Size: {total_size / 1e6:.1f} MB")
print(f"  Location: {OUTPUT_DIR}")

# List saved files
print(f"\nSaved files:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
    print(f"  - {f} ({size:.1f} MB)")

# %% [markdown]
# ## Cell 11: Test the Model

# %%
print("\n" + "=" * 60)
print("TESTING TRAINED MODEL")
print("=" * 60)

def generate_response(prompt, max_new_tokens=100):
    """Generate a response from the trained model"""
    full_prompt = create_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response

# Test with appropriate prompts based on dimension type
if DIMENSION in ["warmth", "playful", "formal", "encouragement"]:
    test_prompts = [
        f"Rewrite this with high {DIMENSION}: I can help you with that.",
        f"Rewrite this with high {DIMENSION}: That's correct.",
        f"Rewrite this with low {DIMENSION}: I'd be happy to assist you.",
    ]
else:
    test_prompts = [
        f"Explain a basic concept in {DIMENSION}.",
        f"What is an important principle in {DIMENSION}?",
    ]

print("\nTest Results:\n")
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    response = generate_response(prompt)
    print(f"Response: {response}\n")
    print("-" * 40)

# %% [markdown]
# ## Cell 12: Done!

# %%
print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)

print(f"""
Dimension trained: {DIMENSION}
Model saved to: {OUTPUT_DIR}

NEXT STEPS:
===========

1. To train another dimension:
   - Change DIMENSION = "{DIMENSION}" to the next one
   - Run all cells again
   
   Dimensions to train:
   [ ] warmth
   [ ] playful  
   [ ] formal
   [ ] encouragement
   [ ] medical
   [ ] coding
   [ ] teaching
   [ ] quantum

2. After ALL dimensions are trained:
   - Download models from Google Drive
   - Run mergekit to combine them
   - Create Clara!

W&B Dashboard:
  https://wandb.ai/chris_hartline/clara-deng-research

Models location:
  Google Drive/Lily/models/
""")

# Finish W&B run
wandb.finish()

print("✓ All done! Change DIMENSION and run again for the next one.")
