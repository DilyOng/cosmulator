"""Tests for the HPO module that do not require optuna or the training stack.

``optuna`` and ``flowjax`` are imported lazily inside :func:`cosmulator.hpo.tune`,
so the module imports and its search-space helper can be exercised without them.
The full study is covered by GPU runs under the ``[train]`` + ``[hpo]`` extras.
"""

from cosmulator import hpo
from cosmulator.hpo import DEFAULT_WARM_START, suggest_hyperparameters


class _FakeTrial:
    """A trial stub that returns the low end of each suggested range."""

    def __init__(self):
        self.asked = []

    def suggest_int(self, name, low, high, log=False):
        self.asked.append(name)
        assert low <= high
        return low

    def suggest_float(self, name, low, high, log=False):
        self.asked.append(name)
        assert low <= high
        return low

    def suggest_categorical(self, name, choices):
        self.asked.append(name)
        assert len(choices) >= 1
        return choices[0]


def test_module_imports_without_optuna():
    assert hasattr(hpo, "tune")


class TestSearchSpace:
    def test_returns_trainer_kwargs(self):
        hp = suggest_hyperparameters(_FakeTrial())
        assert set(hp) == {
            "flow_layers", "nn_width", "nn_depth",
            "learning_rate", "batch_size", "spline",
        }

    def test_all_parameters_are_suggested(self):
        trial = _FakeTrial()
        suggest_hyperparameters(trial)
        assert set(trial.asked) == {
            "flow_layers", "nn_width", "nn_depth",
            "learning_rate", "batch_size", "spline",
        }

    def test_suggested_values_are_in_range(self):
        hp = suggest_hyperparameters(_FakeTrial())
        assert hp["flow_layers"] == 4
        assert hp["nn_depth"] == 1
        assert hp["batch_size"] == 256
        assert hp["spline"] is False
        assert hp["learning_rate"] > 0


class TestWarmStart:
    def test_default_warm_start_matches_the_search_space(self):
        # Every warm-start key must be a real trainer hyperparameter.
        keys = set(suggest_hyperparameters(_FakeTrial()))
        assert set(DEFAULT_WARM_START) == keys

    def test_default_warm_start_is_the_known_good_config(self):
        assert DEFAULT_WARM_START["flow_layers"] == 8
        assert DEFAULT_WARM_START["learning_rate"] == 1e-3
        assert DEFAULT_WARM_START["spline"] is False
