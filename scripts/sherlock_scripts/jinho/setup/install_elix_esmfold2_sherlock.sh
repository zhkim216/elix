#!/usr/bin/env bash
set -euo pipefail

# Build a standalone Torch 2.8 Elix environment with the pinned Biohub
# ESMFold2 stack. The canonical elix-torch280 environment is never modified.

ELIX_USER="${USER:-$(id -un)}"
ELIX_HOME="${HOME:-/home/users/${ELIX_USER}}"
PROJECT_ROOT="${PROJECT_ROOT:-${ELIX_HOME}/code/elix}"
SCRATCH="${SCRATCH:-/scratch/users/${ELIX_USER}}"
VENV="${ELIX_ESMFOLD2_VENV:-${SCRATCH}/envs/uv/elix-esmfold2}"
MANIFEST_ROOT="${ELIX_ESMFOLD2_MANIFEST_ROOT:-${SCRATCH}/debug/elix-esmfold2-install}"
ESM_SOURCE="${ELIX_ESMFOLD2_ESM_SOURCE:-${PROJECT_ROOT}/allatom_design/esm}"
ESM_BASE_REVISION=917af90b624535eed1e072d343c717e3ec11fef4
TRANSFORMERS_REVISION=ef32577f55da19a4989cd7b22e004dc43a4998cb
DOCKQ_REVISION=ba4df5adaad7c77fd60851d0b7b05f2b77061ba2

export PROJECT_ROOT SCRATCH
export ELIX_TORCH280_VENV="${VENV}"
export ELIX_TORCH280_MANIFEST_ROOT="${MANIFEST_ROOT}/elix-base"

bash "${PROJECT_ROOT}/scripts/sherlock_scripts/jinho/setup/install_elix_sherlock.sh"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is required after the base Elix installation" >&2
  exit 1
}

if [[ ! -d "${ESM_SOURCE}" ]]; then
  echo "Editable Biohub ESM source is missing: ${ESM_SOURCE}" >&2
  exit 1
fi
if [[ -e "${ESM_SOURCE}/.git" ]]; then
  git -C "${ESM_SOURCE}" merge-base --is-ancestor \
    "${ESM_BASE_REVISION}" HEAD || {
    echo "ESM checkout is not based on ${ESM_BASE_REVISION}: ${ESM_SOURCE}" >&2
    exit 1
  }
fi

uv pip install --python "${VENV}/bin/python" "transformers @ git+https://github.com/Biohub/transformers.git@${TRANSFORMERS_REVISION}"
uv pip install --python "${VENV}/bin/python" --no-deps --editable "${ESM_SOURCE}"
uv pip install --python "${VENV}/bin/python" accelerate attrs boto3 brotli cloudpathlib dna-features-viewer httpx ipython ipywidgets msgpack-numpy pydssp pygtrie scikit-learn tenacity zstd "dockq @ git+https://github.com/nrontsis/DockQ.git@${DOCKQ_REVISION}"

mkdir -p "${MANIFEST_ROOT}"
VENV="${VENV}" \
ESM_SOURCE="${ESM_SOURCE}" \
ESM_BASE_REVISION="${ESM_BASE_REVISION}" \
TRANSFORMERS_REVISION="${TRANSFORMERS_REVISION}" \
MANIFEST_ROOT="${MANIFEST_ROOT}" \
"${VENV}/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import allatom_design
import atomworks
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder
from esm.models.esmfold2 import ProteinInput
from esm.utils.structure.molecular_complex import MolecularComplexMetadata
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


def direct_url(package: str) -> dict | None:
    text = importlib.metadata.distribution(package).read_text("direct_url.json")
    if not text:
        return None
    return json.loads(text)


def vcs_commit(package: str) -> str | None:
    metadata = direct_url(package)
    if metadata is None:
        return None
    return metadata.get("vcs_info", {}).get("commit_id")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", os.environ["ESM_SOURCE"], *args],
        text=True,
    ).strip()


assert torch.__version__ == "2.8.0+cu126", torch.__version__
esm_direct_url = direct_url("esm")
assert esm_direct_url is not None
assert esm_direct_url.get("dir_info", {}).get("editable") is True
installed_source = Path(
    unquote(urlparse(esm_direct_url["url"]).path)
).resolve()
assert installed_source == Path(os.environ["ESM_SOURCE"]).resolve()
assert vcs_commit("transformers") == os.environ["TRANSFORMERS_REVISION"]
assert importlib.metadata.version("esm") == "3.3.0"
assert ESMFold2InputBuilder is not None
assert ESMFold2Model is not None
assert "source_residue_indices" in ProteinInput.__dataclass_fields__
assert "chain_to_entity" in MolecularComplexMetadata.__dataclass_fields__
esm_has_git = (Path(os.environ["ESM_SOURCE"]) / ".git").exists()
esm_commit = git("rev-parse", "HEAD") if esm_has_git else None
esm_dirty = (
    bool(git("status", "--porcelain", "--untracked-files=no"))
    if esm_has_git
    else None
)
manifest = {
    "venv": os.environ["VENV"],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "esm": {
        "version": importlib.metadata.version("esm"),
        "source": str(installed_source),
        "base_revision": os.environ["ESM_BASE_REVISION"],
        "commit": esm_commit,
        "dirty": esm_dirty,
        "source_mode": "git_checkout" if esm_has_git else "vendored",
        "direct_url": esm_direct_url,
    },
    "transformers": {
        "version": importlib.metadata.version("transformers"),
        "commit": vcs_commit("transformers"),
    },
}
manifest_path = Path(os.environ["MANIFEST_ROOT"]) / "esmfold2.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("venv", os.environ["VENV"])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("esm", importlib.metadata.version("esm"), installed_source, esm_commit)
print("esm_dirty", esm_dirty)
print(
    "transformers",
    importlib.metadata.version("transformers"),
    vcs_commit("transformers"),
)
print("allatom_design", allatom_design.__file__)
print("atomworks", atomworks.__file__)
print("manifest", manifest_path)
PY

echo "Elix ESMFold2 environment complete: ${VENV}"
