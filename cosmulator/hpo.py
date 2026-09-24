"""Hyperparameter optimisation for MAF emulators, with Optuna.

Each Optuna trial builds a flowjax MAF with the trial's hyperparameters, trains it
by weighted maximum likelihood (:func:`cosmulator.maf.train_maf_emulator`), and
returns the held-out **weighted validation NLL** -- which is the forward KL to the
target posterior up to a constant, so minimising it minimises that divergence. No
separate scorer is needed: the training objective *is* the tuning objective.

The study runs **in-process**: the trainer's per-epoch validation NLL is streamed
to the trial via a callback, so Optuna's pruner can stop hopeless trials early
(this is what a subprocess-per-trial design could not do). A shared journal file
lets several workers run the same study in parallel.

optuna is imported lazily; this module needs the ``[train]`` and ``[hpo]`` extras.
"""

# A known-good starting point (the untuned config that already reached ~1% width),
# enqueued as the first trial so the search begins from a sensible place.
DEFAULT_WARM_START = {
    "flow_layers": 8, "nn_width": 50, "nn_depth": 2, "nn_activation": "relu",
    "learning_rate": 1e-3, "batch_size": 1024, "spline": False,
}


def suggest_hyperparameters(trial):
    """Sample one hyperparameter configuration from the search space.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial to draw suggestions from.

    Returns
    -------
    dict
        Keyword arguments for :func:`cosmulator.maf.train_maf_emulator`. The
        space covers flow depth and width, the conditioner depth and activation
        ({relu, tanh, silu, gelu}), the learning rate, the batch size, and
        whether to use spline (NSF) transformers.
    """
    return {
        "flow_layers": trial.suggest_int("flow_layers", 4, 16),
        "nn_width": trial.suggest_int("nn_width", 32, 256, log=True),
        "nn_depth": trial.suggest_int("nn_depth", 1, 4),
        "nn_activation": trial.suggest_categorical(
            "nn_activation", ["relu", "tanh", "silu", "gelu"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        "spline": trial.suggest_categorical("spline", [False, True]),
    }


def tune(
    samples,
    *,
    n_trials=30,
    parameters=None,
    epochs=1500,
    patience=150,
    prune=True,
    warm_start=DEFAULT_WARM_START,
    storage=None,
    study_name=None,
    sampler_seed=42,
    train_seed=0,
    random_startup_trials=5,
    prune_warmup_epochs=200,
):
    """Run an Optuna study tuning a MAF emulator on one posterior.

    Parameters
    ----------
    samples : anesthetic.Samples
        The chain to fit.
    n_trials : int
        Number of trials this call runs (workers sharing a ``storage`` add up).
    parameters : sequence of str, optional
        Parameters to train on; defaults to the sampled cosmological parameters.
    epochs, patience : int
        Passed to the trainer (max epochs and early-stopping patience).
    prune : bool
        If true, stream per-epoch validation NLL to a ``MedianPruner`` and abort
        trials that track worse than the median at the same epoch.
    warm_start : dict or None
        A configuration enqueued as the first trial. Defaults to
        :data:`DEFAULT_WARM_START`; pass ``None`` to disable, or another study's
        ``best_params`` for cross-chain transfer.
    storage : str or optuna storage, optional
        Shared storage (e.g. a journal file) for parallel workers; ``None`` keeps
        the study in memory.
    study_name : str, optional
        Study name, required when using shared ``storage``.
    sampler_seed : int
        Seed for the TPE sampler (which hyperparameters are proposed).
    train_seed : int
        Fixed seed for every trial's training, so trials differ only in their
        hyperparameters (a fair comparison).
    random_startup_trials : int
        Random trials before TPE modelling engages, and completed trials before
        pruning activates -- a safeguard against a bad warm-start locking the
        search into a poor region.
    prune_warmup_epochs : int
        Epochs within a trial before it may be pruned, so a slow starter is not
        killed prematurely.

    Returns
    -------
    optuna.study.Study
        The completed study; ``study.best_params`` / ``study.best_value`` hold
        the best hyperparameters and their validation NLL.
    """
    # Import the submodules explicitly. On some installs ``optuna`` resolves as a
    # namespace package whose top-level attributes (``optuna.samplers``,
    # ``optuna.create_study``, ``optuna.TrialPruned``) are not auto-populated, so a
    # bare ``import optuna`` then ``optuna.samplers`` raises AttributeError.
    import optuna.exceptions
    import optuna.pruners
    import optuna.samplers
    import optuna.study

    from cosmulator.maf import train_maf_emulator

    def objective(trial):
        hp = suggest_hyperparameters(trial)
        reporter = None
        if prune:
            def reporter(epoch, val_nll):
                trial.report(val_nll, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        result = train_maf_emulator(
            samples, parameters=parameters, epochs=epochs, patience=patience,
            seed=train_seed, report=reporter, certify=False, **hp,
        )
        return result["best_val_nll"]

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=random_startup_trials, seed=sampler_seed)
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=random_startup_trials, n_warmup_steps=prune_warmup_epochs)
        if prune else optuna.pruners.NopPruner()
    )
    study = optuna.study.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        storage=storage, study_name=study_name, load_if_exists=True,
    )
    if warm_start is not None:
        study.enqueue_trial(warm_start)
    study.optimize(objective, n_trials=n_trials)
    return study
