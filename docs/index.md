# pySPAC Documentation

pySPAC (python Solar Phase curve Analysis and Characterization) is a library for analyzing and fitting astronomical phase curves of asteroids and other small solar system bodies.

## Repository & Links

- **GitHub Repository**: [https://github.com/rcboufleur/pySPAC](https://github.com/rcboufleur/pySPAC)
- **PyPI Package**: [https://pypi.org/project/pyspac/](https://pypi.org/project/pyspac/)
- **Documentation**: [https://rcboufleur.github.io/pySPAC/](https://rcboufleur.github.io/pySPAC/)
- **Issue Tracker**: [https://github.com/rcboufleur/pySPAC/issues](https://github.com/rcboufleur/pySPAC/issues)

## What pySPAC Does

pySPAC determines the absolute magnitude (H) and other phase function parameters from phase angle (α) and reduced magnitude observations using:

- Standard IAU photometric models: (H, G), (H, G₁, G₂), (H, G₁₂), and linear fit
- Weighted least-squares fitting with observational uncertainties
- Monte Carlo uncertainty analysis
- Model generation from fitted parameters

## Quick Example

```python
import numpy as np
from pyspac import PhaseCurve

# Observational data
angles = np.array([0.17, 0.63, 0.98, 1.62, 4.95, 9.78,
                  12.94, 13.27, 13.81, 17.16, 18.52, 19.4])
mags = np.array([6.911, 7.014, 7.052, 7.105, 7.235, 7.341,
                      7.425, 7.427, 7.437, 7.511, 7.551, 7.599])
errors = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
                   0.02, 0.02, 0.03, 0.03, 0.04, 0.04])

# Create object and fit model
pc = PhaseCurve(angle=angles, magnitude=mags, magnitude_unc=errors)
pc.fitModel(model="HG", method="trust-constr")
pc.summary()

# Monte Carlo uncertainties
pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")
pc.summary()
```

## Installation

```bash
pip install pyspac
```

## How to Cite

If you use pySPAC in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17163112.svg)](https://doi.org/10.5281/zenodo.17163112)

```bibtex
@software{boufleur_pyspac_2025,
  author = {Boufleur, Rodrigo Carlos},
  title = {pySPAC: Solar Phase curve Analysis and Characterization},
  url = {https://github.com/rcboufleur/pySPAC},
  doi = {10.5281/zenodo.17163112},
  version = {0.1.0},
  year = {2025}
}
```

Additionally, please cite the relevant papers for the models you employ:

- **HG System**: Bowell, E., et al. (1989). "Application of photometric models to asteroids." In Asteroids II (pp. 524-556).
- **HG1G2 & HG12 Systems**: Muinonen, K., et al. (2010). "A three-parameter magnitude phase function for asteroids." Icarus, 209(2), 542-555.
- **HG12PEN System**: Penttilä, A., et al. (2016). "Asteroid H, G1, G2 and H, G12 phase function performance with sparse data." Planetary and Space Science, 123, 117-122.


## Navigation

- **[Getting Started](getting-started.md)** - Basic workflow
- **[Setting Up Objects](setup-objects.md)** - Creating PhaseCurve objects
- **[Fitting Models](fitting-models.md)** - Model fitting procedures
- **[Setting Boundaries](boundaries.md)** - Parameter constraints
- **[Models and Methods](models-methods.md)** - Available models and fitting methods
- **[Monte Carlo Uncertainties](uncertainties.md)** - Error estimation
- **[Save and Load](save-load.md)** - Data persistence
- **[Plotting Results](plotting.md)** - Visualization
- **[Generate Models](generate-models.md)** - Synthetic phase curves
