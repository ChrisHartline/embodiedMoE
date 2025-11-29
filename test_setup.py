"""
Test that everything is set up correctly
"""

import os
from dotenv import load_dotenv

def test_environment():
    """Test Python environment and packages"""
    print("=" * 60)
    print("TESTING ENVIRONMENT")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing package imports...")
    try:
        import numpy as np
        print("   ✓ numpy")
    except ImportError:
        print("   ✗ numpy - run: pip install numpy")
        return False
    
    try:
        import torch
        print(f"   ✓ torch (version {torch.__version__})")
        print(f"     CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("   ✗ torch - run: pip install torch")
        return False
    
    try:
        import transformers
        print(f"   ✓ transformers (version {transformers.__version__})")
    except ImportError:
        print("   ✗ transformers - run: pip install transformers")
        return False
    
    try:
        from anthropic import Anthropic
        print("   ✓ anthropic")
    except ImportError:
        print("   ✗ anthropic - run: pip install anthropic")
        return False
    
    try:
        from dotenv import load_dotenv
        print("   ✓ python-dotenv")
    except ImportError:
        print("   ✗ python-dotenv - run: pip install python-dotenv")
        return False
    
    # Test .env file
    print("\n2. Testing .env file...")
    load_dotenv()
    
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key:
        print(f"   ✓ CLAUDE_API_KEY found (length: {len(api_key)})")
    else:
        print("   ✗ CLAUDE_API_KEY not found in .env")
        print("     Create a .env file with: CLAUDE_API_KEY=your_key_here")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_environment()
    
    if success:
        print("\nYou're ready to run the demos!")
        print("\nNext steps:")
        print("  1. Run: python 1_hdc_basics.py")
        print("  2. Run: python 3_tiny_model.py")
        print("  3. Run: python 4_clara_prototype.py")
        print("  4. Run: python 2_personality_data.py (uses API key)")
    else:
        print("\nPlease fix the issues above and run again.")