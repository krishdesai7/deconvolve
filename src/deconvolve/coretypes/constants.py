from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final, LiteralString


# The one floating type the pipeline carries, end to end.
#
# The jet inputs justify it: `mass` and `mult` are bit-exact through a float32
# round trip, and the other observables lose at most half a ULP, so there is
# no structure below float32 to preserve. 320 paired seeds put float32 and
# float64 within 3.5 sigma of each other on unfolding improvement -- see
# `benchmarks/precision.py`. Everything
# downstream -- the containers, the models, `JAX_ENABLE_X64` -- follows from
# this line.
EVENT_DTYPE: Final[type[np.single]] = np.single

# Everything RAN can regenerate lives under one root: the dataset `.npz`
# caches and the XLA compilation cache. `DECONVOLVE_CACHE_DIR` relocates the whole
# tree, which a cluster needs (on Perlmutter `$HOME` is small and quota'd;
# `$SCRATCH` is not). This is deliberately not derived from `XDG_CACHE_HOME`.
#
# Read once, at import, since the `cache_dir=` defaults throughout `deconvolve.data`
# bind to this module-level constant at import either way.
CACHE_ENV_VAR: Final[LiteralString] = "DECONVOLVE_CACHE_DIR"
CACHE_DIR: Final[Path] = Path(
    os.environ.get(key=CACHE_ENV_VAR) or ".cache"
).expanduser()

# XLA keys its persistent cache on lowered HLO plus the jaxlib and backend
# versions, so a stale entry is a miss rather than a wrong answer -- upgrading
# JAX or changing an architecture costs a recompile, never a wrong number.
#
# It is unbounded by default (`jax_compilation_cache_max_size` is -1), but a run
# adds only a few MB and only for a shape/config it has not seen, so it plateaus
# rather than grows. Set that config if a shared directory needs a ceiling; it
# also turns on the file lock, which is otherwise absent.
COMPILE_CACHE_DIR: Final[Path] = CACHE_DIR / "jax"

RUN_DIR: Final[Path] = Path("runs")

# Every distance metric is *displayed* multiplied by this -- in the report, the
# terminal table and the leakage-check log alike -- and *stored* unscaled.
# `metrics.json` is a machine interface, so it holds the divergences as
# defined; raw, a real twelve-observable run spans 6.2e-3..3.0e-1
# (Wasserstein) and 8.5e-5..1.3e-2 (JS), which is what a human reader needs
# scaled.
METRIC_DISPLAY_SCALE: Final[float] = 1e3

# A run directory is read by people. `config.json` and `report.pdf` stay at the
# root because they are what a person opens; everything else -- checkpoints,
# arrays, figures, the metrics and timing JSON -- is supporting material and
# lives one level down, flat.
ARTIFACTS_DIR: Final[LiteralString] = "artifacts"


def artifacts_dir(run_dir: Path, /) -> Path:
    """The run's supporting-material subdirectory, created on demand."""
    path: Path = run_dir / ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


ZENODO_RECORD: Final[int] = 3548091
GENERATORS: Final[tuple[LiteralString, LiteralString]] = ("Pythia26", "Herwig")
N_FILES: Final[int] = 17
# A tuple, emphatically not a `frozenset`: these names select *columns*, and
# `load_jet_dataset` fills column `i` from the `i`-th name, so the container
# holding them is an ordering and a set has none. The order here matches
# `JET_OBS` below.
SUBSTRUCTURE_VARIABLES: Final[tuple[LiteralString, ...]] = (
    "m",
    "M",
    "w",
    "tau21",
    "zg",
    "sdm",
    # Beyond the OmniFold six. `--var` selects any subset, so the original
    # configuration stays reachable as `--var m --var M --var w --var tau21
    # --var zg --var sdm`.
    "q",
    "f_ch",
    "lha",
    "ang2",
    "ptd",
    "n_ch",
)

# Cache-safe filenames: avoid case collisions on case-insensitive filesystems
# (macOS APFS default), where "m.npz" and "M.npz" resolve to the same path.
CACHE_FILENAMES: Final[dict[str, str]] = {
    "m": "mass",
    "M": "mult",
    "w": "w",
    "tau21": "tau21",
    "zg": "zg",
    "sdm": "sdm",
    "q": "q",
    "f_ch": "f_ch",
    "lha": "lha",
    "ang2": "ang2",
    "ptd": "ptd",
    "n_ch": "n_ch",
}


class JetVarInfo(NamedTuple):
    xlim: tuple[float, float]
    xlabel: str
    symbol: str


# The value `_get_var` writes for a jet groomed to nothing, where ln(rho^2) is
# undefined. It is **not** a bound on the observable: real jets reach -37.9, so
# this sentinel sits inside the distribution rather than below it, and ~0.75% of
# events fall past it. `SDM_XLIM` is a separate number for that reason -- an
# axis limit chosen for where the bulk lives, not derived from the sentinel.
#
# The degenerate jets are a spike superimposed on a smooth tail, and the
# fraction is generator-dependent (detector level: Herwig 0.034%, Pythia
# 0.057%) -- exactly the shape of thing `benchmarks/response.py` is built to
# detect. But the most information an "is it at the floor?" bit can carry at
# those rates is 1.5e-5 nats, against a measured I(S; X | Z) of 3.6e-3 --
# 0.42% of the effect. Moving the sentinel would shift the standardization
# statistics for a correction two orders of magnitude below what it would fix.
#
# The spike is never ambiguous, either: reaching exactly -14.0 from a
# continuous log is measure-zero, so an event at the sentinel is a degenerate
# jet and nothing else.
LOG_RHO_FLOOR: Final[float] = -14.0

# Covers ~99.25% of events. The remainder is a genuine tail, not an artifact.
SDM_XLIM: Final[tuple[float, float]] = (-14.0, -2.0)
JET_OBS: Final[dict[str, JetVarInfo]] = {
    "m": JetVarInfo(xlim=(0, 75), xlabel="Jet Mass", symbol=r"$m$ [GeV]"),
    "M": JetVarInfo(xlim=(0, 80), xlabel="Jet Constituent Multiplicity", symbol=r"$M$"),
    "w": JetVarInfo(xlim=(0, 0.6), xlabel="Jet Width", symbol=r"$w$"),
    "tau21": JetVarInfo(
        xlim=(0, 1.2),
        xlabel=r"$N$-subjettiness Ratio",
        symbol=r"$\tau_{21}^{(\beta=1)}$",
    ),
    "zg": JetVarInfo(
        xlim=(0, 0.5), xlabel="Groomed Jet Momentum Fraction", symbol=r"$z_g$"
    ),
    "sdm": JetVarInfo(xlim=SDM_XLIM, xlabel="Soft Drop Jet Mass", symbol=r"$\ln\rho$"),
    "q": JetVarInfo(xlim=(-0.5, 0.5), xlabel="Jet Charge", symbol=r"$q$"),
    "f_ch": JetVarInfo(xlim=(0, 1), xlabel="Jet Charge Fraction", symbol=r"$f_{ch}$"),
    # `w` is the angularity lambda^1_1, so `lha` (beta = 1/2) and `ang2`
    # (beta = 2) complete the family around it. `ang2` is close to
    # m^2 / (pT R)^2, which is why it is the most promising of these for
    # constraining the jet-mass response.
    "lha": JetVarInfo(
        xlim=(0, 0.8), xlabel="Les Houches Angularity", symbol=r"$\lambda^{1}_{0.5}$"
    ),
    "ang2": JetVarInfo(
        xlim=(0, 0.3), xlabel="Jet Angularity", symbol=r"$\lambda^{1}_{2}$"
    ),
    "ptd": JetVarInfo(
        xlim=(0.1, 1), xlabel="Transverse Momentum Dispersion", symbol=r"$p_T^D$"
    ),
    "n_ch": JetVarInfo(
        xlim=(0, 50), xlabel="Charged Constituent Multiplicity", symbol=r"$n_{ch}$"
    ),
}

# How the observables are *presented*. This is not `SUBSTRUCTURE_VARIABLES`,
# and must never become it: that tuple is the column order and the cache key.
# The column order carries no physics; this
# one does, and is applied at render time only.
#
# m -> ln rho -> lambda^1_0.5 -> w -> lambda^1_2 -> z_g -> tau_21
#   -> M -> n_ch -> f_ch -> p_T^D -> q
#
# `M` next to `n_ch` exposes the baseline hadronization ratio (n_ch / M ~ 2/3,
# from pion isospin); `f_ch` bridges particle counting and track-based energy
# reconstruction; `p_T^D` completes the quark/gluon discriminant system with
# `M` and `n_ch`; `q` closes as the valence flavour indicator.
JET_DISPLAY_ORDER: Final[tuple[LiteralString, ...]] = (
    "m",
    "sdm",
    "lha",
    "w",
    "ang2",
    "zg",
    "tau21",
    "M",
    "n_ch",
    "f_ch",
    "ptd",
    "q",
)

JET_VARIABLE_GROUPS: Final[tuple[tuple[str, tuple[LiteralString, ...]], ...]] = (
    ("Mass and hard scale (IRC-safe kinematics)", ("m", "sdm")),
    ("Continuous angularities (IRC-safe jet shapes)", ("lha", "w", "ang2")),
    ("Splitting and 2-prong substructure", ("zg", "tau21")),
    (
        "Hadronization, multiplicity and fragmentation (IRC-unsafe)",
        ("M", "n_ch", "f_ch", "ptd", "q"),
    ),
)


# The level figures' page layout.
#
# A panel's width on the rendered page is `linewidth / PANEL_COLUMNS`
# regardless of the figure's own inch size, because `\includegraphics` scales
# the whole figure by exactly as much as widening it grew the figure -- so
# the column count alone sets panel width. The figure's *absolute* inches,
# unaffected by that scaling, instead set the rendered text size:
# `font.size * linewidth_pt / (72 * figure_width_in)`. So `PANEL_COLUMNS`
# sizes the panels and `PANEL_WIDTH_INCHES` sizes their labels, independently.
#
# A hist-over-ratio cell reads best WIDER than tall, roughly 5:4. At 3
# columns against a 749.4pt landscape text block, `PANEL_WIDTH_INCHES = 7.0`
# renders the 18pt base font at 8.9pt, a normal figure text size in print;
# paired with `_LevelStyle.height_per_dim = 6.6`, a page of six spans 83% of
# the block's height. `PANELS_PER_PAGE = 6` (3x2) is the largest grid that
# keeps panels 5:4-ish without either shrinking them (2x2, more pages) or
# splitting twelve observables awkwardly (3x3, a 9+3 page pair).
#
# `report.py` needs these same numbers to know how many `\includegraphics`
# pages to emit, and must stay free of matplotlib, so they live here rather
# than in `plotting`.
PANEL_COLUMNS: Final[int] = 3
PANELS_PER_PAGE: Final[int] = 6
PANEL_WIDTH_INCHES: Final[float] = 7.0


def figure_pages(dim: int, /) -> int:
    """How many pages a level figure spans for `dim` observables."""
    return max(1, math.ceil(dim / PANELS_PER_PAGE))


def display_order(variables: Sequence[str], /) -> tuple[int, ...]:
    """Indices into `variables`, reordered for presentation.

    Filters `JET_DISPLAY_ORDER` to what this run actually holds, so a `--var`
    subset stays in physics order. A non-jet run (`dim_0`, `dim_1`, ...) has no
    entry in the table and falls through to the identity.
    """
    position: dict[str, int] = {name: i for i, name in enumerate(iterable=variables)}
    ordered: tuple[int, ...] = tuple(
        position[name] for name in JET_DISPLAY_ORDER if name in position
    )
    return ordered if len(ordered) == len(variables) else tuple(range(len(variables)))


DEFAULT_PURITY_THRESHOLD: Final[np.double] = np.sqrt(0.5)
TRUTH_SENTINEL: Final[np.double] = np.double(np.iinfo(int_type=np.short).min)

# What `deconvolve leakage-check --poison` overwrites z_true with. Any far-off-manifold
# value does the job, so this is only a default -- but it must not be
# TRUTH_SENTINEL. A truth column set entirely to that value is exactly what
# `Populations.create` writes when there is no truth at all, so `has_truth`
# would report the poisoned arm as having none and `require_truth()` would
# refuse the particle-level comparison the check exists to make.
POISON_SENTINEL: Final[np.double] = np.double(-999.0)

LOG2: Final[float] = np.log(2.0)
