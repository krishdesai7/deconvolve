<!-- markdownlint-disable no-inline-html -->
# Reporting & Artifacts

<span style="font-variant: small-caps;">Deconvolve</span> produces self-contained run directories containing models, training histories, evaluated metrics, and compiled LaTeX/PDF reports.

---

## Run Directory Layout

Each execution of `deconvolve train` creates a timestamped run directory under `runs/` with the following structure:

```shell
runs/2026-09-19-164500_tag/
├── config.json          # Complete serialized run configuration
├── train_history.json   # Per-epoch loss and validation metrics
├── models/
│   ├── generator.keras      # Best generator checkpoint
│   └── discriminator.keras  # Best discriminator checkpoint
├── metrics.json         # Evaluated distance metrics (via `deconvolve evaluate`)
└── artifacts/           # Generated figures and compiled LaTeX reports
    ├── loss_curves.png
    ├── mmd_trajectory.png
    ├── pull_plots.png
    ├── report.tex
    └── report.pdf
```

---

## Automated LaTeX Dossiers

Running `deconvolve report` compiles a comprehensive report dossier. E.g.,

```shell
deconvolve report runs/2026-09-19-164500
```

### Generated Content

- **Configuration Summary**: Complete record of hyperparameters, dataset specifications, and seeds.
- **Metric Comparison Table**: Unweighted vs. Reweighted Wasserstein-1 distances, Jensen-Shannon divergences, and triangular discriminators across all dimensions.
- **Pull & Ratio Plots**: Ratio of reweighted simulation to data across feature distributions with statistical error bars.
- **Training Dynamics**: Loss trajectories for $g$ and $d$, demonstrating convergence to the theoretical equilibrium $\log 2 \approx 0.693$.
- **MMD Model Selection**: Validation MMD curve indicating the selected checkpoint epoch.

---

## Programmatic Access

You can inspect run histories and metrics programmatically in Python:

```python
import json
from pathlib import Path

run_dir = Path("runs/2026-09-19-164500")

with (run_dir / "metrics.json").open() as f:
    metrics = json.load(f)

print("Reweighted Wasserstein-1:", metrics["reweighted"]["wasserstein_1d"])
```
