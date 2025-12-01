"""
Complete Clara Monitoring System with Weights & Biases Integration
Tracks: data generation, fine-tuning, merging, evaluation, HDC memory

Usage:
    from clara_monitoring_complete import ClaraMonitor
    
    monitor = ClaraMonitor(project_name="clara-deng-research")
    monitor.track_data_generation(dimension="warmth", ...)
    monitor.track_merge_experiment(merge_name="v1", ...)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dotenv import load_dotenv

import numpy as np

# Load environment variables
load_dotenv()

# Import wandb (with graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")


class ClaraMonitor:
    """
    Complete monitoring system for Clara development pipeline
    Integrates with Weights & Biases for comprehensive experiment tracking
    """
    
    def __init__(
        self,
        project_name: str = "clara-deng-research",
        entity: Optional[str] = "chris_hartline",  # Add your entity as default
        api_key: Optional[str] = None,
        offline_mode: bool = False
    ):
        """
        Initialize Clara monitoring system
        
        Args:
            project_name: W&B project name
            entity: W&B username/team (optional, uses default)
            api_key: W&B API key (or uses WB env variable)
            offline_mode: If True, logs locally without syncing to cloud
        """
        self.project_name = project_name
        self.entity = entity
        self.offline_mode = offline_mode
        self.current_run = None
        
        # Setup directories for local logging
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        
        if not WANDB_AVAILABLE:
            print("⚠️  Running without W&B - logs will be saved locally only")
            return
        
        # Get API key from: parameter > WB env var > WANDB_API_KEY env var
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        elif 'WB' in os.environ:
            os.environ['WANDB_API_KEY'] = os.environ['WB']
        
        # Set offline mode if requested
        if offline_mode:
            os.environ['WANDB_MODE'] = 'offline'
        
        # Login to W&B
        try:
            wandb.login(key=os.environ.get('WANDB_API_KEY'))
            self.logged_in = True
            print("✓ Logged into Weights & Biases")
        except Exception as e:
            print(f"⚠️  W&B login issue: {e}")
            print("   Run: wandb login")
            self.logged_in = False