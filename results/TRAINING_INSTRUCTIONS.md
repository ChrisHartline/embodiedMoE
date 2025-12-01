
# Clara Fine-tuning Instructions

## Data Files (upload to Google Drive)
- ./data/warmth_training.json
- ./data/playful_training.json
- ./data/formal_training.json
- ./data/encouragement_training.json
- ./data/medical_knowledge.json
- ./data/coding_knowledge.json
- ./data/teaching_knowledge.json
- ./data/quantum_knowledge.json

## Training Steps

1. Upload all data files to: Google Drive/clara_data/

2. For each dimension/domain, run training in Colab:
   - Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
   - Method: LoRA (r=16, alpha=32)
   - Epochs: 3
   - Batch size: 4 (effective 16 with grad accumulation)

3. Save models to: Google Drive/clara_models/

4. Download to local: ./models/tinyllama_<dimension>/

## Colab Notebook Template

See: ./configs/clara_training_template.ipynb
