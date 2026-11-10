# Changelog — NAS V6

- Added a bounded meta-operator grammar with eight registered primitives.
- Added safe PyTorch compilation of declarative operator graphs.
- Added error-guided operator graph proposal and graph-level insert/delete/replace/reorder mutation.
- Integrated meta-operator graphs into composed NAS branches and generation-zero innovation proposals.
- Updated search metadata and search-mode label to V6.
- Added tests for forward/backward execution, graph mutation, invalid primitive rejection, and search integration.
- Validation: `python -m unittest discover -s tests_agent -v` — 55 tests passed.
- ETTh1 meta-operator CPU training attempt timed out; no V6 ETTh1 performance claim is made.
- Model promotion remains disabled pending independent evaluation and release gates.
