@echo off
echo ===================================================
echo Setting up Voice Studio Pro Environment
echo ===================================================

:: 1. We must install PyTorch FIRST to establish the core CUDA 12.8 backend. 
:: If this isn't installed first, subsequent audio packages will fail to find the GPU bindings.
echo [1/4] Installing PyTorch (CUDA 12.8)...
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

:: 2. Install Flash Attention from the local wheel you downloaded from the Wildminder repo.
:: NOTE: Update the filename below to match the exact .whl file you downloaded for your Python version!
echo [2/4] Installing Flash Attention 2...
uv pip install flash_attn-2.8.2+cu128torch2.7cxx11abiFALSE-cp310-cp310-win_amd64.whl

:: 3. Install Transformers, strictly pinned to 4.57.3 to maintain compatibility with Qwen3-TTS decorators.
echo [3/4] Installing Hugging Face Transformers...
uv pip install "transformers[audio]==4.57.3"

:: 4. Force-install all remaining dependencies from the requirements file to catch anything left over.
echo [4/4] Installing remaining dependencies...
uv pip install -r requirements.txt --reinstall

echo ===================================================
echo Installation Complete! You can now launch the app.
echo ===================================================
pause