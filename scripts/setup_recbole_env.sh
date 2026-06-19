#!/usr/bin/env bash
# Isolated environment for RecBole / SASRec, so it can't perturb the main book_recsys env.
# RecBole pins torch + numpy/scipy versions that we don't want leaking into the project env.
# Run from the repo root:  bash scripts/setup_recbole_env.sh
# Then in Jupyter/VS Code pick the "book_recsys (recbole)" kernel for notebooks/06_recbole.ipynb.
set -euo pipefail

ENV=".venv-recbole"
PY="${PYTHON:-python3}"

"$PY" -m venv "$ENV"
# shellcheck disable=SC1091
source "$ENV/bin/activate"
python -m pip install -U pip wheel

# book_recsys (editable) → importable in this env regardless of the notebook's CWD,
# so 06's bootstrap cell's `import book_recsys.data` just works (no Kaggle clone path).
python -m pip install -e .

# The SASRec stack. recbole pulls torch (CPU/MPS wheel on macOS); kmeans-pytorch is an
# optional recbole dep pip won't pull on its own.
python -m pip install recbole kmeans-pytorch

# So the notebook can run on this env.
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name book-recsys-recbole \
    --display-name "book_recsys (recbole)"

# Smoke test: everything imports together.
python - <<'PY'
import book_recsys.data, book_recsys.models.content.maxsim  # our code
import recbole, torch, numpy
print("OK — recbole", recbole.__version__, "| torch", torch.__version__,
      "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())
PY

echo
echo "Done. Activate with:  source $ENV/bin/activate"
echo "In Jupyter/VS Code, select the 'book_recsys (recbole)' kernel for 06_recbole.ipynb."
