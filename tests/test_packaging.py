"""Tests that the declared packaging metadata matches what the code needs.

Dependency metadata drifts silently: an import is added without a corresponding
declaration, and the package keeps working only because some other dependency
happens to pull the module in. ``pip check`` does not catch this, because it
compares declared metadata against what is installed rather than imports against
declarations. These tests close that gap in both directions.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "cosmulator"

#: Import name -> distribution name, where the two differ.
IMPORT_TO_DISTRIBUTION = {"yaml": "pyyaml"}

#: Modules supplied by an optional extra rather than the core dependencies.
#: Anything listed here must be imported lazily, inside a function, so that
#: importing cosmulator never requires the training stack.
OPTIONAL_IMPORTS = {"margarine", "jax", "flax", "optax"}


def _module_paths():
    return sorted(PACKAGE.rglob("*.py"))


def _top_level_imports(tree):
    """Yield the top-level module name of every absolute import in a tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, i.e. our own package.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _third_party_imports():
    """Non-stdlib modules imported anywhere in the package."""
    modules = set()
    for path in _module_paths():
        modules.update(_top_level_imports(ast.parse(path.read_text())))
    return {
        m for m in modules if m not in sys.stdlib_module_names and m != "cosmulator"
    }


def _declared_dependencies():
    """Runtime dependency names from [project] dependencies in pyproject.toml."""
    text = (REPO_ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert block, "no [project] dependencies array found in pyproject.toml"
    return {
        re.split(r"[<>=!~\[; ]", item)[0].strip().lower()
        for item in re.findall(r'"([^"]+)"', block.group(1))
    }


def test_every_imported_package_is_declared():
    """Each third-party import is a declared dependency or a known extra."""
    declared = _declared_dependencies()
    missing = sorted(
        m
        for m in _third_party_imports()
        if m not in OPTIONAL_IMPORTS
        and IMPORT_TO_DISTRIBUTION.get(m, m).lower() not in declared
    )
    assert not missing, (
        f"imported but not declared in pyproject.toml: {missing}. "
        f"Declared: {sorted(declared)}"
    )


def test_declared_dependencies_are_all_used():
    """No runtime dependency is declared without being imported."""
    imported = {
        IMPORT_TO_DISTRIBUTION.get(m, m).lower() for m in _third_party_imports()
    }
    unused = sorted(d for d in _declared_dependencies() if d not in imported)
    assert not unused, f"declared in pyproject.toml but never imported: {unused}"


def test_optional_stack_is_not_imported_at_module_scope():
    """The training stack must not be imported at import time.

    ``pip install cosmulator`` deliberately omits JAX so that the package stays
    installable and reviewable without a GPU. That guarantee only holds if every
    optional import happens inside a function or method.
    """
    offenders = []
    for path in _module_paths():
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in _top_level_imports(
                    ast.Module(body=[node], type_ignores=[])
                ):
                    if name in OPTIONAL_IMPORTS:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
    assert not offenders, (
        "optional training dependencies imported at module scope; import them "
        f"inside the function that needs them instead: {offenders}"
    )
