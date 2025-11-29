@echo off
echo Installing Clara dependencies...

echo.
echo Step 1: Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo Step 2: Installing other dependencies...
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install transformers>=4.40.0
pip install datasets>=2.14.0
pip install accelerate>=0.27.0
pip install sentencepiece>=0.1.99
pip install anthropic>=0.28.0
pip install python-dotenv>=1.0.0
pip install tqdm>=4.65.0

echo.
echo Installation complete!
echo Run: python test_setup.py