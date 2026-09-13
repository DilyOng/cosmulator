# Contributing to cosmulator

Contributions are welcome, whether that is a bug report, a documentation fix or
a new feature.

## Reporting problems

Open an issue at https://github.com/DilyOng/cosmulator/issues. For a bug, please
include the version of `cosmulator`, your Python version, and the smallest
example that reproduces the problem.

## Proposing changes

For anything larger than a small fix, please open an issue first so the approach
can be agreed before you spend time on it.

## Running the checks locally

The checks below are exactly what CI runs, so a clean run locally should mean a
green pull request.

```bash
pip install -e ".[test]"

pytest                                    # test suite
flake8 cosmulator tests                   # settings in .flake8
pydocstyle --convention=numpy cosmulator  # numpydoc docstrings
black --check cosmulator tests            # formatting
isort --check-only cosmulator tests       # import order
```

## Conventions

- Docstrings follow the numpydoc convention.
- Line length is 88 characters, matching `black`.
- The training stack (JAX, margarine) is an **optional** dependency. It must
  never be imported at module scope, so that `pip install cosmulator` stays
  usable without a GPU. `tests/test_packaging.py` enforces this.
- Every third-party import must be declared in `pyproject.toml`, and every
  declared dependency must actually be imported. This is also enforced by
  `tests/test_packaging.py`.

## Support

This is research software maintained alongside a PhD, so responses may take a
little time. Issues are the best way to reach the maintainer.
