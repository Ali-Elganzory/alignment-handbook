uv venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip

uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

uv pip install -e .

uv pip install "flash-attn==2.7.4.post1" --no-build-isolation