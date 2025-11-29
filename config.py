"""
Configuration for Clara prototype
"""

# HDC Settings
HDC_DIMENSIONS = 10000

# Model Settings
TINY_MODEL_NAME = "roneneldan/TinyStories-33M"  # or "distilgpt2"

# Personality Defaults
DEFAULT_PERSONALITY = {
    'warmth': 0.8,
    'formality': 0.3,
    'verbosity': 0.6,
    'encouragement': 0.9
}

# Data Generation
CLAUDE_API_KEY = None  # Set via environment variable
TRAINING_EXAMPLES_PER_DIMENSION = 1000

# Paths
DATA_DIR = "./data"
MODELS_DIR = "./models"