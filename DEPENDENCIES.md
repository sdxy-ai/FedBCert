# Dependencies

Bootstrap installation with `pip>=25.1,<26`, setuptools 79.0.1 and wheel
0.48.0 as shown in the README. Older pip versions can reject normalized
dependency metadata from the CPU wheel index. These installation tools are
separate from the numerical runtime pins and original environment inventory.

The validated RG-1.0 environment is Linux/WSL2 with Python **3.9.25** and
CPU-only PyTorch **2.8.0+cpu**. Package metadata permits Python 3.9–3.11;
this does not mean that every interpreter/platform combination was tested.
Native Windows is unsupported by historical replay paths using POSIX `resource`.

| Package | Version | Purpose |
| --- | --- | --- |
| PyTorch | 2.8.0; validated wheel 2.8.0+cpu | Models and sparse propagation |
| PyTorch Geometric | 2.6.1 | Graph data and benchmark utilities |
| NumPy | 2.0.0 | Generation and paired statistics |
| SciPy | 1.13.1 | Sparse/scientific utilities |
| scikit-learn | 1.6.1 | Splits and metrics |
| Matplotlib | 3.9.2 | Historical result plots |
| OGB | 1.3.6 | Optional ogbn-arxiv access |
| PyMetis | 2023.1.1 | Graph partitioning |
| PyYAML | 6.0.3 | Configuration |
| psutil | 7.2.2 | Resource measurements |
| pytest | 8.4.2 | Regression tests |

Install the CPU PyTorch wheel first, then `requirements-no-torch.txt`, followed
by `python -m pip install -e . --no-deps`, as shown in the README. The full
`requirements.txt` also names PyTorch. Run `python -m pip check` after installing.
The offline review paths do not require a GPU, Gurobi, `torch_sparse` or
`torch_scatter`; they use native PyTorch sparse COO operations.

`environment/requirements.lock.txt` records all installed distributions from
the frozen RG-1.0 environment, including transitive dependencies and bootstrap
tools. `environment/requirements.wheels-hashed.txt` records resolved wheel
hashes. These files document that specific environment, not a universal
cross-platform installation recipe: retained wheels and Python bootstrapping
are not bundled. Root requirements are the package installation entry point.
`environment/public-review-requirements.lock.txt` records the separate public
release validation environment; its installation tools differ from the frozen
experiment environment, while the numerical package versions are unchanged.
Do not silently upgrade numerical packages when comparing saved statistics.

## Optional comparisons

**FedLap+ (Arnoldi):** `bash scripts/setup_fedlap.sh` fetches the official
repository at its pinned revision. Install
`python -m pip install -r requirements-optional-fedlap.txt` (pandas 2.3.3 and
python-dotenv 1.2.1), together with the relevant upstream requirements.
FedLap+ is a variant from the original paper, not a new method introduced here.
Its teacher uses stronger global structural information than the verifier.

**GNNev:** `bash scripts/setup_gnnev.sh` fetches the pinned official code and
installs `gurobipy==12.0.3` into a local compatibility directory. A valid Gurobi
license is required. These solver checks are marked `optional` and excluded
from the default review command.

External repositories, datasets and license credentials are not bundled.
Installation needs network access; installed offline smoke, toy inference
and saved-record checks do not.
