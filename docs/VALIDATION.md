# Release checks

Release 1.2.0 was checked on October 21, 2026 using Linux/WSL2, Python 3.9.25,
PyTorch 2.8.0+cpu, PyG 2.6.1, NumPy 2.0.0 and pytest 8.4.2.

A new virtual environment was created without system-site packages. The check
verified imported package paths and versions, a single NumPy distribution,
native sparse COO multiplication, editable installation and `pip check`.
The installed inventory is in
[`environment/public-review-requirements.lock.txt`](../environment/public-review-requirements.lock.txt).
Numerical dependencies were installed from 58 retained wheels whose SHA-256
hashes match the frozen wheel inventory, rather than copied from an installed
environment. Installation used `--no-index --find-links` for those wheels and
`--no-build-isolation` for the local editable package. The README describes
the network installation route; this check is not a fresh network download of
every dependency. The initial pip 23.0.1 CPU-index attempt failed on normalized
dependency metadata; the README now bootstraps pip 25.3 before installation.

| Check | Result |
| --- | --- |
| All seven quick-check scripts in the README | Passed |
| Portable pytest selection | **511 passed, 1 skipped, 11 deselected** |
| Same commands from a separate copy outside the parent project | Passed; same test counts |
| Scientific-source comparison | 160 unchanged pinned files; three record-helper adaptations explicitly indexed |
| RG-1.0 saved statistics | 32 cases, 1,536 records and 512 case medians reproduced |
| Historical synthetic on-demand statistics | 240 measurements reproduced; 22.09% paired payload reduction and 23.76% CPU increase |

The skipped module needs optional GNNev/Gurobi dependencies. The 11 deselected
tests require the omitted parent-project runner archive. Fourteen warnings
come from Matplotlib's use of deprecated Pyparsing interfaces; no assertion or
dependency check failed. Tests include tiny synthetic optimization steps, not
benchmark training. No historical model inference or timing rerun was performed.

The three public helper adaptations resolve old source pins to preserved
pre-fix copies. They do not change algorithms, saved records or statistical
formulas; original helper bodies and hashes are retained. Current package
hashes and historical bindings were checked separately. See
[`PROVENANCE.md`](PROVENANCE.md).

These checks establish package integrity and finite implementation consistency,
not strict floating-point soundness or an audit of omitted graph/checkpoint
archives. The full RG-1.0 training archive remains outside this compact release.

[`ci-tests.yml.example`](ci-tests.yml.example) provides an optional Python 3.10
CPU workflow. It is not active or claimed to have passed; the results above
are local Python 3.9.25 checks.
