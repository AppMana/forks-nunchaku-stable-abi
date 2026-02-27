#!/bin/bash
set -e

echo "=== Setting up stable ABI test environments ==="

# Base venv for building (Python 3.12 + torch 2.9)
echo "Creating build venv (Python 3.12 + Torch 2.9)..."
uv venv .venv-build --python 3.12
uv pip install --python .venv-build torch==2.9 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-build ninja setuptools wheel build numpy

# Cross-Python venv (Python 3.13, install wheel built by 3.12)
echo "Creating cross-Python venv (Python 3.13)..."
uv venv .venv-py313 --python 3.13
uv pip install --python .venv-py313 torch==2.9 --index-url https://download.pytorch.org/whl/cu128

# Cross-Torch venv (Python 3.12, torch 2.10)
echo "Creating cross-Torch venv (Python 3.12 + Torch 2.10)..."
uv venv .venv-torch210 --python 3.12
uv pip install --python .venv-torch210 torch==2.10 --index-url https://download.pytorch.org/whl/cu128

echo "=== All test environments created ==="
echo ""
echo "Next steps:"
echo "  1. Build:        source .venv-build/bin/activate && NUNCHAKU_INSTALL_MODE=FAST python -m build --wheel --no-isolation"
echo "  2. Smoke test:   source .venv-build/bin/activate && python -c 'from nunchaku._C import QuantizedFluxModel; print(\"OK\")'"
echo "  3. Cross-Python: source .venv-py313/bin/activate && pip install dist/nunchaku-*-cp39-abi3-*.whl && python -c 'from nunchaku._C import QuantizedFluxModel; print(\"OK\")'"
echo "  4. Cross-Torch:  source .venv-torch210/bin/activate && pip install dist/nunchaku-*-cp39-abi3-*.whl && python -c 'from nunchaku._C import QuantizedFluxModel; print(\"OK\")'"
