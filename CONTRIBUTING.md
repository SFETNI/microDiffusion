Please avoid committing large notebook outputs (images, long arrays) to keep the repository small.

Recommended contributor steps:

- Install `nbstripout` (recommended):

  ```bash
  pip install nbstripout
  nbstripout --install
  ```

- Or use `pre-commit` (preferred):

  ```bash
  pip install pre-commit
  pre-commit install
  ```

- If you need to remove outputs from a notebook you've already edited:

  ```bash
  nbstripout path/to/notebook.ipynb
  # or
  pre-commit run --all-files
  ```

Why: Notebook outputs (embedded PNG/base64) inflate the repository and make clones slower. Using the hooks above prevents accidental commits of heavy outputs.
