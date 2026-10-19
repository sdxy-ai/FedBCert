# Dependencies

Versions below match the submitted code. Release checks use Python 3.9.25 on
Linux/WSL2, CPU execution, and PyTorch 2.8.0+cu128 installed. Package metadata
supports Python 3.9–3.11; retraining numerics may differ across platforms.

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 2.8.0 | Models, sparse propagation and certificates |
| PyTorch Geometric | 2.6.1 | Graph data and benchmark utilities |
| NumPy | 2.0.0 | Generation and paired statistics |
| SciPy | 1.13.1 | Scientific utilities |
| scikit-learn | 1.6.1 | Splits and metrics |
| Matplotlib | 3.9.2 | Figures |
| OGB | 1.3.6 | ogbn-arxiv access |
| PyMetis | 2023.1.1 | Partitions |
| PyYAML | 6.0.3 | Configuration |
| psutil | 7.2.2 | Resource measurements |
| pytest | >=8,<9 | Tests |

`requirements.txt` and `pyproject.toml` declare these dependencies. Install the
platform-specific PyTorch wheel first, then `requirements-no-torch.txt` as in
the README. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for platform details. Core checks do not need optional compiled PyG extensions,
Gurobi or a GPU.

## Optional comparisons

**FedLap+ (Arnoldi):** `bash scripts/setup_fedlap.sh` fetches the official
repository at the pinned revision. Install
`python -m pip install -r requirements-optional-fedlap.txt` (pandas 2.3.3,
python-dotenv 1.2.1), plus any remaining upstream requirements. FedLap+ is the
variant in the original paper, not a new algorithm introduced here. Its teacher
uses stronger global structural information than the endpoint-blind verifier.

**GNNev:** `bash scripts/setup_gnnev.sh` fetches the pinned official code and
installs `gurobipy==12.0.3` into a local compatibility directory. A valid Gurobi
license is required. External solver tests are excluded from the quick suite.

External repositories, datasets and license credentials are not bundled.
Installation needs network access; installed reference checks do not.
