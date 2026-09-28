# Vectorized Histogram Metrics Design

## Goal

Refactor the private histogram metric helpers in `ran.evaluate` so normalized
histograms are represented as two matrices and downstream divergences operate
across all feature dimensions with NumPy and SciPy axis operations.

## Histogram helper contract

`_normalized_histograms(ref, comp, weights, n_bins)` will return a tuple
`(p, q)`. Both arrays have shape `(dimensions, n_bins)` and floating-point
values. A one-dimensional input is treated as one feature and therefore
produces arrays with shape `(1, n_bins)`.

Each feature retains its own uniform bin edges spanning the combined minimum
and maximum of the corresponding `ref` and `comp` values. `p` is unweighted;
`weights`, when provided, applies only to `q`. Each nonzero histogram is
normalized independently. An all-zero histogram remains all zero, preserving
the existing behavior.

The helper will continue to call `numpy.histogram` once per feature because
each feature has distinct bin edges. This keeps NumPy's established bin-edge
semantics and avoids a custom flattened-bin implementation.

## Metric consumers

`_js_per_dim` will pass the two histogram matrices to
`scipy.spatial.distance.jensenshannon` with `axis=1`, square the returned
distances, and return the resulting one-dimensional array.

`_triangular_per_dim` will calculate the squared differences and denominators
for both matrices at once. A masked `numpy.divide` will contribute zero where
the denominator is zero, and the result will be summed over the bin axis and
scaled by `1e3` as before.

No public evaluation output or metric definition changes.

## Compatibility and errors

Plain one-dimensional arrays and two-dimensional sample-by-feature arrays are
supported. Existing weighting semantics and failures for otherwise malformed
inputs remain unchanged; broader input validation is outside this refactor.

## Testing

Focused tests will establish the new helper contract before production changes:

- one-dimensional inputs return `(1, n_bins)` matrices and no longer fail from
  two-dimensional indexing;
- multidimensional inputs produce one normalized histogram row per feature;
- comparison weights affect only the comparison histogram;
- Jensen-Shannon and triangular results match independently calculated
  per-feature formulas and return one value per feature.

The focused tests and the complete project test suite will be run after the
refactor. Static formatting and lint checks will be run on changed files.
