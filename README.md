# llments

## Development Information

To start developing, install the development dependencies and pre-commit hooks:

```bash
pip install ".[dev]"
pre-commit install
```

There are several [pre-commit hooks](https://pre-commit.com/) that will run on every
commit to perform formatting, typing, and linting.

* `ruff` - Runs formatting, import sorting, and linting.
* `mypy` - Runs type checking.
* `markdownlint` - Runs markdown linting.
* `yamllint` - Runs YAML linting.
