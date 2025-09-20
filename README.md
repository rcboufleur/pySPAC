# pySPAC: Solar Phase curve Analysis and Characterization

pySPAC (python Solar Phase curve Analysis and Characterization) is a library for analyzing and fitting astronomical phase curves of asteroids and other small solar system bodies. It provides an object-oriented interface to bridge observational photometric data with standard theoretical models.

The primary function of pySPAC is to determine the absolute magnitude (H) and other phase function parameters from a set of phase angle (α) and reduced magnitude observations.

## Documentation

📖 Full documentation is available at: https://rcboufleur.github.io/pySPAC/

## Features
- Object-oriented interface via the PhaseCurve class
- Standard IAU photometric models: (H, G), (H, G₁, G₂), (H, G₁₂), (H, G₁₂) Penttilä calibration, and linear fit
- Weighted least-squares fitting with observational uncertainties
- Monte Carlo uncertainty analysis
- Multiple optimization methods with automatic constraint validation
- Model generation from fitted parameters
- Data persistence - save and load object state to/from JSON
- Comprehensive error handling and validation

## Installation

### From PyPI
```bash
pip install pyspac
```

### From GitHub
```bash
pip install git+https://github.com/rcboufleur/pySPAC.git
```

## Quick Start
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

For detailed examples and tutorials, see the full documentation.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| LINEAR | H, β | Linear phase function |
| HG | H, G | IAU standard (H, G) system |
| HG12 | H, G₁₂ | Simplified (H, G₁₂) system |
| HG12PEN | H, G₁₂ | Penttilä et al. (2016) calibration |
| HG1G2 | H, G₁, G₂ | Three-parameter (H, G₁, G₂) system |

## Model-Method Compatibility
- **Constrained models** (HG, HG12, HG12PEN, HG1G2): Must use `trust-constr`, `SLSQP`, or `COBYLA`
- **Unconstrained models** (LINEAR): Can use any optimization method

pySPAC automatically validates model-method compatibility and provides clear error messages.

## Monte Carlo Uncertainty Analysis

Two Monte Carlo methods for different scientific cases:

- `.monteCarloUncertainty()`: Based on observational uncertainties
- `.monteCarloUnknownRotation()`: Includes rotational scatter modeling

## Data Requirements

| Model | Min Points | Recommended |
|-------|------------|-------------|
| All 2-parameter models | 3 | 5+ |
| HG1G2 (3-parameter) | 3* | 5+ |

*Due to mathematical constraint 1-G₁-G₂=0

## Contributing

Contributions are welcome! Please:

1. Fork the repository on GitHub
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Submit a pull request to the main branch

Feel free to open an issue to report bugs or suggest new features.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
