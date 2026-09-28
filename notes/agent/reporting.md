# Reporting

`deconvolve report <run_dir>` compiles one run directory into a PDF dossier at
`<run_dir>/report.pdf` — configuration, timings, the metric tables, and
every figure, in one document. `--force` rebuilds over an existing PDF;
`--no-compile` stops at `artifacts/report.tex` so the LaTeX can be inspected
without a TeX installation.

**The run directory has two files at the top and everything else under
`artifacts/`.** `config.json` and `report.pdf` are what a human opens; the
weights, histories, metric JSON and component figures are inputs to the
report, and burying them is what made the directory readable at a glance.
`artifacts_dir(run_dir)` is the single accessor and it creates the directory,
so anything that only _reads_ must use `run_dir / ARTIFACTS_DIR` instead —
rendering a report must not mkdir into a directory it was handed.

`src/deconvolve/reporting/templates/report.tex` is the document; `src/deconvolve/reporting/report.py` only fills
in `<<TOKEN>>` slots and never decides layout. All rounding policy lives in
the template's `siunitx` column types, so changing how a number reads is a
LaTeX edit, not a Python one.

Two things about the figures are load-bearing:

**Jet observables are presented in physics order, not storage order.**
`JET_DISPLAY_ORDER` and `JET_VARIABLE_GROUPS` in `coretypes/constants.py` give
the twelve observables as four physics groups (mass and hard scale;
continuous angularities; splitting and 2-prong substructure; hadronization,
multiplicity and fragmentation), and `display_order()` maps a run's variables
onto it. This is presentation only — `SUBSTRUCTURE_VARIABLES` remains the
canonical storage order and the cache key, and must not be reordered (see
[data-model.md](data-model.md)).

**The level figures are paginated, and the report has to agree.**
`figure_pages(dim)` is the page count for both, six panels to a page;
`report.py` emits that many `\includegraphics[page=k]` blocks without opening
the file. A run whose figures were drawn before pagination has a one-page PDF
and `pdflatex` fails with "required page does not exist" — redraw with
`deconvolve train --load-run <run_dir>` first. `submit.zsh` keeps them in step.

The figure pages are landscape with their own `\newgeometry{margin=8mm}`,
and two independent knobs set how they read. A panel's width on the page is
`linewidth / PANEL_COLUMNS` whatever the figure's inch size, because
`\includegraphics[width=\linewidth]` scales the figure by exactly as much as
widening it grew the figure. What the inches DO set is the rendered text size,
`font.size * linewidth_pt / (72 * figure_width_in)`. So `PANEL_COLUMNS` sizes
the panels and `PANEL_WIDTH_INCHES` sizes their labels, downwards.

Figure defects do not fail tests. Clipped labels, missing titles, overlapping
text, an occluded inset and a wrong panel aspect all passed a green suite here
and were caught only by rendering the PDF and measuring artist bounding boxes.
Render and look before claiming a plotting change works.
