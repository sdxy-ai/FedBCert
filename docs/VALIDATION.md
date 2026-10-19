# Release checks

Local validation on October 19, 2026 used Linux/WSL2, Python 3.9.25,
PyTorch 2.8.0+cu128 with CPU execution, PyG 2.6.1, NumPy 2.0.0 and pytest 8.4.2.
The isolated project environment reused installed dependencies; it was not a
fresh download of every dependency. Editable installation and `pip check` passed.

All five quick-check scripts listed in the README passed. The portable pytest
suite reported **265 passed, 1 skipped**. No benchmark training, historical
model inference or timing rerun was performed.

The saved-record checker reproduced 240 measurements, a 22.09% mean per-seed
payload reduction, and a 23.76% CPU increase (23.8% rounded to one decimal place).
It did not re-audit the omitted graph/checkpoint archive.

All 148 model, protocol and result files compared with the submitted archives
were byte-identical. Staged Git objects were also compared with the release hash
index before upload. This establishes the published source mapping, not the
mathematical validity of every possible completion.

`docs/ci-tests.yml.example` provides the same checks for a fresh Python 3.10 CPU
environment. It is not an active workflow: the publishing credential does not
have GitHub's workflow-write permission. A repository administrator can enable
it later. No CI pass is claimed; the results above are local checks.
