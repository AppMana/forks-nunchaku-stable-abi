# Stable ABI testing targets

.PHONY: setup-test-envs test-build test-import test-cross-python test-cross-torch test-all clean-test-envs

setup-test-envs:
	bash scripts/test_stable_abi.sh

test-build:
	. .venv-build/bin/activate && NUNCHAKU_INSTALL_MODE=FAST python -m build --wheel --no-isolation

test-import:
	. .venv-build/bin/activate && python -c "\
		from nunchaku._C import QuantizedFluxModel, QuantizedSanaModel, QuantizedGEMM, QuantizedGEMM88; \
		m = QuantizedFluxModel(); \
		print('All classes instantiated OK'); \
		import torch; \
		print(f'torch.ops.nunchaku available: {hasattr(torch.ops, \"nunchaku\")}'); \
		print('Import test PASSED')"

test-cross-python:
	. .venv-py313/bin/activate && \
		pip install dist/nunchaku-*-cp39-abi3-linux_x86_64.whl && \
		python -c "from nunchaku._C import QuantizedFluxModel; print('Cross-Python (3.13) test PASSED')"

test-cross-torch:
	. .venv-torch210/bin/activate && \
		pip install dist/nunchaku-*-cp39-abi3-linux_x86_64.whl && \
		python -c "from nunchaku._C import QuantizedFluxModel; print('Cross-Torch (2.10) test PASSED')"

test-all: test-build test-import test-cross-python test-cross-torch
	@echo "All stable ABI tests passed!"

clean-test-envs:
	rm -rf .venv-build .venv-py313 .venv-torch210
