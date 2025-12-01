import json
from pathlib import Path

# Check what's in the data directory
data_dir = Path('./data')
print("Files in ./data:")
for f in data_dir.glob('*'):
    print(f"  {f.name} - {f.stat().st_size / 1024:.1f} KB")

# Load and check warmth data
warmth_file = data_dir / 'warmth_training.json'
if warmth_file.exists():
    with open(warmth_file) as f:
        warmth_data = json.load(f)
    
    print(f"\nWarmth data: {len(warmth_data)} examples")
    
    if warmth_data:
        print("\nFirst 2 examples:")
        for i, ex in enumerate(warmth_data[:2], 1):
            print(f"\n{i}.")
            print(f"  Neutral: {ex.get('neutral', 'N/A')}")
            print(f"  Low:     {ex.get('low', 'N/A')}")
            print(f"  High:    {ex.get('high', 'N/A')}")
else:
    print("\nNo warmth_training.json found!")