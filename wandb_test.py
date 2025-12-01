import wandb
import os
from dotenv import load_dotenv

load_dotenv()

# Check if key is loaded
api_key = os.environ.get('WANDB_API_KEY') or os.environ.get('WB')
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"Key starts with: {api_key[:10]}..." if api_key else "No key")

# Try to login and create a test run
try:
    wandb.login(key=api_key)
    
    run = wandb.init(
        entity="chris_hartline",
        project="clara-deng-research",
        name="test-connection",
        job_type="test"
    )
    
    wandb.log({"test_metric": 1.0})
    
    print(f"\n✓ SUCCESS!")
    print(f"Dashboard: https://wandb.ai/chris_hartline/clara-deng-research")
    print(f"Run URL: {run.url}")
    
    run.finish()
    
except Exception as e:
    print(f"\n❌ Error: {e}")