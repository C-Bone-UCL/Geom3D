# List all commands.
default:
  @just --list

# Build docs.
docs:
  rm -rf docs/source/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/build/html/index.html

# Do a dev install.
dev_gpu:
  pip install numpy
  pip install ipykernel
  pip install rdkit
  pip install scipy
  pip install psutil
  pip install wandb
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu118
  pip install torch_geometric
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  pip install cython
  pip install -e '.[dev]'

dev_cpu:
  pip install numpy
  pip install rdkit
  pip install ipykernel
  pip install scipy
  pip install psutil
  pip install wandb
  python3 -m pip install torch
  pip install torch_geometric
  pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
  pip install cython
  pip install -e '.[dev]'


# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff . )

  echo
  ( set -x; black --check . )

  echo
  ( set -x; mypy . )

  echo
  ( set -x; pytest --cov=src --cov-report term-missing )

  echo
  ( set -x; make -C docs doctest )

  test $error = 0

# Auto-fix code issues.
fix:
  black .
  ruff --fix .

# Build a release.
build:
  python -m build
