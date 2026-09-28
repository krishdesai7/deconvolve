<!-- markdownlint-disable no-inline-html -->
# Datasets

<span style="font-variant: small-caps;">Deconvolve</span> supports two dataset regimes, selected with `--dataset` (`-D`): synthetic **Gaussian datasets** (`gaussian`, the default for `train`) described by a YAML file, and **jet substructure observables** (`jets`) downloaded from Zenodo.

!!! note
    The Gaussian YAML file passed with `--config` describes a *dataset*. It is unrelated to `deconvolve.toml`, which sets default *CLI options*; see [Configuration](configuration.md).

---

## Gaussian Datasets (YAML)

Gaussian toy datasets specify the mean and covariance for generation, nature (truth), and detector smearing.

### File Format

A Gaussian configuration file is a YAML document containing:

```yaml
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9          # scalar, vector, or full matrix
sigma_true: 1.0
sigma_detector: 0.5
```

### Sigma-to-Covariance Promotion

<span style="font-variant: small-caps;">Deconvolve</span> automatically promotes `sigma_*` entries into `(dim, dim)` positive-definite covariance matrices:

- **Scalar** \(\sigma \to \sigma^2 I\) (isotropic diagonal covariance)
- **Vector** \(\lbrack \sigma_1, \sigma_2, \dots \rbrack \to \text{diag}(\sigma_1^2, \sigma_2^2, \dots)\) (uncorrelated diagonal covariance)
- **Matrix** \(\Sigma\) used as-is (full correlated covariance)

Positive-definiteness is verified via Cholesky decomposition during configuration parsing.

### Multidimensional Examples

Pre-configured examples are provided in the `params/` directory:

=== "1D (`params/1d_default.yaml`)"
    ```yaml
    mu_gen: [0.5]
    mu_true: [0.0]
    sigma_gen: 0.9
    sigma_true: 1.0
    sigma_detector: 0.5
    ```

=== "2D Correlated (`params/2d_correlated.yaml`)"
    ```yaml
    mu_gen: [0.5, -0.3]
    mu_true: [0.0, 0.0]
    sigma_gen:
      - [0.9, 0.2]
      - [0.2, 0.8]
    sigma_true:
      - [1.0, 0.0]
      - [0.0, 1.0]
    sigma_detector: 0.5
    ```

=== "4D Correlated (`params/4d_correlated.yaml`)"
    ```yaml
    mu_gen: [1.0, 0.0, -0.5, 0.5]
    mu_true: [0.8, 0.1, -0.6, 0.7]
    sigma_gen:
      - [ 1.0,   0.07, -0.22,  0.24 ]
      - [ 0.07,  0.49,  0.0,   0.056]
      - [-0.22,  0.0,   1.21,  0.616]
      - [ 0.24,  0.056, 0.616, 0.64 ]
    sigma_true:
      - [ 0.64,  0.0,  -0.24,  0.192]
      - [ 0.0,   0.36,  0.12,  0.0  ]
      - [-0.24,  0.12,  1.0,   0.3  ]
      - [ 0.192, 0.0,   0.3,   0.36 ]
    sigma_detector: [0.4, 0.5, 0.6, 0.3]
    ```

=== "6D Correlated (`params/6d_correlated.yaml`)"
    ```yaml
    mu_gen: [1.0, 0.0, -0.5, 0.5, -1.0, 0.3]
    mu_true: [0.8, 0.1, -0.6, 0.7, -0.8, 0.1]
    sigma_gen:
      - [ 1.0,   0.07,  0.22, -0.24,  0.0,   0.0  ]
      - [ 0.07,  0.49,  0.0,  -0.112, 0.252, 0.098]
      - [ 0.22,  0.0,   1.21,  0.088,-0.264, 0.462]
      - [-0.24, -0.112, 0.088, 0.64,  0.096, 0.0  ]
      - [ 0.0,   0.252,-0.264, 0.096, 1.44,  1.176]
      - [ 0.0,   0.098, 0.462, 0.0,   1.176, 1.96 ]
    sigma_true:
      - [ 0.64,  0.0,   0.16, -0.096, 0.08,  0.0  ]
      - [ 0.0,   0.36,  0.0,  -0.036, 0.12,  0.0  ]
      - [ 0.16,  0.0,   1.0,   0.0,  -0.3,   0.44 ]
      - [-0.096,-0.036, 0.0,   0.36,  0.12,  0.0  ]
      - [ 0.08,  0.12, -0.3,   0.12,  1.0,   0.55 ]
      - [ 0.0,   0.0,   0.44,  0.0,   0.55,  1.21 ]
    sigma_detector: [0.4, 0.5, 0.6, 0.3, 0.4, 0.4]
    ```

---

## Jet Substructure Observables

Jet substructure observables are selected with `--var` (`-v`):

```shell
deconvolve train -Djets -vm -vw
```

`--var` (`-v`) is repeatable. With no `--var`, all twelve observables are used. The order typed in does not matter. E.g., `-vw -vm` and `-vm -vw` are equivalent.

The data is derived from [Zenodo record 3548091](https://zenodo.org/record/3548091): $Z+$jets events with $p_T^Z > 200$ GeV and [<span style="font-variant: small-caps;">Delphes</span>](https://github.com/delphes/delphes) detector simulation. <span style="font-variant: small-caps;">Herwig</span> plays the role of data and <span style="font-variant: small-caps;">Pythia26</span> the role of simulation. Each observable is available at both particle (nominal) level and detector (reconstructed) level.

!!! warning
    Names are case-sensitive: `m` is the jet mass, `M` the constituent multiplicity.

### Mass and hard scale

| Name | Symbol | Observable | Source |
| :--- | :--- | :--- | :--- |
| `m` | \(m\) | Jet mass | Directly from the release |
| `sdm` | \(\ln\rho\) | Soft Drop jet mass, \(\ln\left(\frac{m_{\text{SD}}^2}{p_T^2}\right)\) | Derived: release provides the groomed mass and jet \(p_T\); `-14.0` for a jet groomed to nothing |

### Continuous angularities

| Name | Symbol | Observable | Source |
| :--- | :--- | :--- | :--- |
| `lha` | \(\lambda^1_{0.5}\) | Les Houches angularity | Directly from the release |
| `w` | \(w = \lambda^1_1\) | Jet width | Directly from the release |
| `ang2` | \(\lambda^1_2\) | Angularity with \(\beta = 2\) | Directly from the release |

### Splitting and 2-prong substructure

| Name | Symbol | Observable | Source |
| :--- | :--- | :--- | :--- |
| `zg` | \(z_g\) | Soft Drop groomed momentum fraction | Directly from the release |
| `tau21` | \(\tau_{21}^{(\beta=1)}\) | \(N\)-subjettiness ratio \(\frac{\tau_2}{\tau_1}\) | Derived: release provides \(\tau_2\) and \(w = \tau_1^{(\beta=1)}\); `0` for a zero-width jet |

### Hadronization, multiplicity and fragmentation (IRC-unsafe)

| Name | Symbol | Observable | Source |
| :--- | :--- | :--- | :--- |
| `M` | \(M\) | Constituent multiplicity | Directly from the release |
| `n_ch` | \(n_{ch}\) | Charged constituent multiplicity | Computed from constituents |
| `f_ch` | \(f_{ch}\) | Charged fraction of the jet's constituent \(p_T\) | Computed from constituents |
| `ptd` | \(p_T^D\) | \(p_T\) dispersion, \(\frac{\sqrt{\sum p_{T,i}^2}}{\sum p_{T,i}}\) | Computed from constituents |
| `q` | \(q\) | Jet charge, \(p_T-\)weighted with \(\kappa = \frac12\) | Computed from constituents |

!!! tip
    The original OmniFold study used the observables `m`, `M`, `w`, `tau21`, `zg`, `sdm`. To use only those:
    ```shell
    deconvolve train -Djets -vm -vM -vw -vtau21 -vzg -vsdm
    ```
