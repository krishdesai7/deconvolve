# ruff: file-ignore[module-import-not-at-top-of-file]
"""Split each run's generator-weight variance into visible and invisible parts.

    Var(w) = Var(E[w|x]) + E[Var(w|x)]
           =  visible    +  residual (what the detector averages away)

E[w|x] is fit by an MLP regressing w(z_gen) on x_sim over the MC train split, and
the split is measured on the held-out MC test split. Alongside each run, two
reference rows come from classifiers (evaluation only -- never fed to a generator):

    oracle w*(z)  ~ p_truth(z) / p_gen(z)    classifier on z_true vs z_gen
    omega(x)      ~ p_data(x)  / q_sim(x)    classifier on x_data vs x_sim

`reco gap` is E[(h - omega)^2] on MC test: how far each run's reweighted sim is
from data at detector level.

Run from the repo root, e.g.:

uv run benchmarks/dispersion_split.py runs/<jets λ=0.015> runs/<jets λ=0>\
    --out split.json

All runs passed together must share a dataset (same dataset, n_samples,
variables, data_seed): the references are fit once, on the first run's data.
Calibration on the 1D Gaussian (analytic / binned answers in brackets):
the generator rows reproduce them to <0.005 -- λ=0 0.483 = 0.332 + 0.156
[0.487 = 0.330 + 0.157], λ=0.015 0.272 = 0.196 + 0.077 [0.274 = 0.196 + 0.077].
The classifier references do NOT: oracle Var 0.431 [0.547], omega 0.292
[0.350]. Classifier density ratios shrink the tails, so read the reference
rows as underestimates of the true spread.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

_ = os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
from deconvolve.evaluation.evaluate import _generator_weights, _load_splits
from deconvolve.training.models import load_model, set_random_seed

if TYPE_CHECKING:
    from collections.abc import Callable

    from deconvolve.coretypes.events import Populations

SHARED_KEYS = (
    "dataset",
    "n_samples",
    "dim",
    "variables",
    "data_seed",
    "gaussian_params",
)


def _mlp(dim: int, out_activation: str, width: int, depth: int) -> keras.Model:
    inputs: keras.KerasTensor = keras.Input(shape=(dim,))  # pyrefly: ignore[unknown-attribute-access]
    h: keras.KerasTensor = inputs
    for _ in range(depth):
        h = keras.layers.Dense(width, activation="relu")(h)
    out: keras.KerasTensor = keras.layers.Dense(1, activation=out_activation)(h)
    return keras.Model(inputs, out)


def _fit(
    model: keras.Model,
    x: np.ndarray,
    t: np.ndarray,
    loss: str,
    args: argparse.Namespace,
) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=loss)
    model.fit(
        x,
        t,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        shuffle=True,
        verbose=2 if args.verbose else 0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3),
        ],
    )
    return model


def _predict(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(
        model.predict(x, batch_size=batch_size, verbose=0), dtype=np.float64
    ).ravel()


class Standardizer:
    def __init__(self, ref: np.ndarray) -> None:
        self.mu = ref.mean(axis=0)
        self.sd = ref.std(axis=0) + 1e-8

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.mu) / self.sd).astype(np.float32)


def _ratio(
    num_train: np.ndarray, den_train: np.ndarray, args: argparse.Namespace
) -> Callable[[np.ndarray], np.ndarray]:
    """p_num / p_den via a balanced classifier, as a function of den-sample rows
    (mean 1 on each)."""
    std = Standardizer(den_train)
    x = np.concatenate([std(num_train), std(den_train)])
    y = np.concatenate([np.ones(len(num_train)), np.zeros(len(den_train))]).astype(
        np.float32
    )
    # Balance the classes so the odds are the density ratio up to a constant.
    sw = np.where(y == 1, len(x) / (2 * len(num_train)), len(x) / (2 * len(den_train)))
    model = _mlp(x.shape[1], "sigmoid", args.width, args.depth)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy")
    model.fit(
        x,
        y,
        sample_weight=sw.astype(np.float32),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        shuffle=True,
        verbose=2 if args.verbose else 0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3),
        ],
    )

    def ratio(rows: np.ndarray) -> np.ndarray:
        p = np.clip(_predict(model, std(rows), args.batch_size), 1e-6, 1 - 1e-6)
        r = p / (1 - p)
        return r / r.mean()

    return ratio


def _split(
    w_train: np.ndarray,
    w_test: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    omega_test: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Fit h(x) = E[w|x] on train, decompose Var(w) on test."""
    w_train = w_train / w_train.mean()
    w_test = w_test / w_test.mean()
    std = Standardizer(x_train)
    h_model = _fit(
        _mlp(x_train.shape[1], "softplus", args.width, args.depth),
        std(x_train),
        w_train.astype(np.float32),
        "mse",
        args,
    )
    h = _predict(h_model, std(x_test), args.batch_size)
    total = float(w_test.var())
    visible = float(h.var())
    residual = float(np.mean((w_test - h) ** 2))
    out = {
        "var_w": total,
        "visible": visible,
        "residual": residual,
        # Should be ~0 when h is a good estimate of E[w|x]; a large value means
        # h under- or over-fit and the two parts are not to be trusted.
        "closure": total - visible - residual,
        "ess_frac": float(1.0 / (1.0 + total)),
    }
    if omega_test is not None:
        out["reco_gap"] = float(np.mean((h - omega_test) ** 2))
    return out


def _populations(config: dict[str, Any]) -> tuple[Populations, Populations]:
    splits = _load_splits(config)
    return splits.train.as_arrays().partition(), splits.test.as_arrays().partition()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = ap.add_argument("runs", nargs="+", type=Path)
    _ = ap.add_argument("--out", type=Path)
    _ = ap.add_argument("--width", type=int, default=128)
    _ = ap.add_argument("--depth", type=int, default=3)
    _ = ap.add_argument("--epochs", type=int, default=60)
    _ = ap.add_argument("--batch-size", type=int, default=4096)
    _ = ap.add_argument("--lr", type=float, default=1e-3)
    _ = ap.add_argument("--seed", type=int, default=0)
    _ = ap.add_argument(
        "--no-references", action="store_true", help="skip the oracle and omega rows"
    )
    _ = ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _load_configs(runs: list[Path]) -> list[dict[str, Any]]:
    configs = [json.loads((r / "config.json").read_text()) for r in runs]
    for r, c in zip(runs[1:], configs[1:], strict=False):
        diff = [k for k in SHARED_KEYS if c.get(k) != configs[0].get(k)]
        if diff:
            raise SystemExit(
                f"{r} differs from {runs[0]} in {diff}; pass them separately"
            )
    return configs


def _reference_rows(
    train: Populations, test: Populations, args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """The omega(x) and oracle w*(z) rows, plus omega on MC test for `reco_gap`."""
    xs_tr, xs_te = train.mc.x, test.mc.x
    print("fitting omega(x) = p_data/q_sim ...", flush=True)
    omega_test = _ratio(train.data, xs_tr, args)(xs_te)
    print("fitting oracle w*(z) = p_truth/p_gen ...", flush=True)
    # z_true enters here and only here, to build the reference weights.
    w_star = _ratio(train.require_truth(), train.mc.z, args)
    rows = {
        "omega(x)": {"visible": float(omega_test.var())},
        "oracle w*": _split(
            w_star(train.mc.z), w_star(test.mc.z), xs_tr, xs_te, omega_test, args
        ),
    }
    return omega_test, rows


def _run_row(
    run: Path,
    config: dict[str, Any],
    train: Populations,
    test: Populations,
    omega_test: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[str, dict[str, float]]:
    lam = config.get("lambda_dispersion")
    print(f"{run.name}: λ={lam} ...", flush=True)
    g = load_model(run / "artifacts" / "generator.keras")
    w_tr = np.asarray(_generator_weights(g, z_gen=train.mc.z), dtype=np.float64)
    w_te = np.asarray(_generator_weights(g, z_gen=test.mc.z), dtype=np.float64)
    row = _split(w_tr, w_te, train.mc.x, test.mc.x, omega_test, args)
    return f"{run.name} (λ={lam})", row


def _print_table(results: dict[str, dict[str, float]]) -> None:
    cols = ("var_w", "visible", "residual", "closure", "reco_gap", "ess_frac")
    name_w = max(len(k) for k in results) + 2
    print("\n" + "".ljust(name_w) + "".join(c.rjust(10) for c in cols))
    for name, row in results.items():
        print(
            name.ljust(name_w)
            + "".join((f"{row[c]:10.4f}" if c in row else "".rjust(10)) for c in cols)
        )


def main() -> None:
    args = _parse_args()
    set_random_seed(args.seed)
    configs = _load_configs(args.runs)

    print(f"loading {configs[0]['dataset']} data ...", flush=True)
    train, test = _populations(configs[0])

    omega_test = None
    results: dict[str, dict[str, float]] = {}
    if not args.no_references:
        omega_test, results = _reference_rows(train, test, args)

    for run, config in zip(args.runs, configs, strict=False):
        label, row = _run_row(run, config, train, test, omega_test, args)
        results[label] = row

    _print_table(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
