# pySPAC: Solar Phase curve Analysis and Characterization

[![PyPI version](https://badge.fury.io/py/pyspac.svg)](https://badge.fury.io/py/pyspac)
[![Build Status](https://img.shields.io/github/actions/workflow/status/rcboufleur/pySPAC/python-package.yml?branch=main)](https://github.com/rcboufleur/pySPAC/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pySPAC` (**py**thon **S**olar **P**hase curve **A**nalysis and **C**haracterization) is a library for analyzing and fitting astronomical phase curves of asteroids and other small solar system bodies. It provides a high-level, object-oriented interface to bridge observational photometric data with standard theoretical models. The library is built on the robust numerical frameworks of `lmfit`, `sbpy`, and `numpy` to ensure reliable and accurate results.

The primary goal of `pySPAC` is to determine the absolute magnitude ($H$) and other key phase function parameters from a set of phase angle ($\alpha$) and reduced magnitude observations, thereby characterizing the object's surface properties.

---

## Installation

You can install `pySPAC` directly from PyPI or from the GitHub repository.

### From PyPI

To get the latest stable release, use `pip`:

```bash
pip install pyspac
```

### From GitHub

To install the latest development version directly from the source code:

```bash
pip install git+https://github.com/rcboufleur/pySPAC.git
```

## The Photometric Models: A Deeper Look

`pySPAC` supports several widely-used photometric systems. Choosing the correct model is crucial and depends on the quality of your data and the physical properties of the object being studied.

### The IAU (H, G) System

The (H, G) system was the standard model adopted by the IAU and remains widely used for legacy applications. It describes the phase curve using the absolute magnitude (H) and a single slope parameter (G) that broadly characterizes the opposition effect.

**Best Used For:** Legacy applications, general-purpose fitting on well-observed, moderate-albedo asteroids (e.g., S- and C-types), and situations requiring compatibility with historical data. It can perform poorly for objects with very high or very low albedos.

**Reference:** Bowell, E., Hapke, B., Domingue, D., et al. (1989). "Application of photometric models to asteroids." In Asteroids II (pp. 524-556). University of Arizona Press.

### The (H, G₁, G₂) System

Developed as a more physically-based alternative to the (H, G) system, the (H, G₁, G₂) model better describes the shape of the phase curve, including the sharpness and amplitude of the opposition effect. The G₁ and G₂ parameters are related to the surface properties of the body. This system has become increasingly adopted in modern asteroid photometry research.

**Best Used For:** High-quality datasets with good coverage of the opposition peak. It is considered a significant improvement and is applicable to a wider range of asteroid types than the (H, G) system. Widely used in contemporary research.

**Reference:** Muinonen, K., Belskaya, I. N., Cellino, A., et al. (2010). "A three-parameter magnitude phase function for asteroids." Icarus, 209(2), 542-555.

### The (H, G₁₂) System

The (H, G₁₂) system is a simplified, two-parameter version of the (H, G₁, G₂) model. It was introduced for cases where the observational data is too sparse to constrain all three parameters effectively. The G₁₂ parameter simultaneously describes the slope and the opposition effect.

**Best Used For:** Sparse datasets (e.g., from all-sky surveys) where fitting the full (H, G₁, G₂) model is not feasible.

**Reference:** Muinonen, K., Belskaya, I. N., Cellino, A., et al. (2010). "A three-parameter magnitude phase function for asteroids." Icarus, 209(2), 542-555.

### The (H, G₁₂) System - Penttilä et al. (2016) calibration (HG12PEN)

This is a recalibration of the (H, G₁₂) model by Penttilä et al. (2016). The basis functions were slightly modified to improve the model's performance, particularly for sparse data.

**Best Used For:** The same sparse-data applications as the standard (H, G₁₂) model, but may offer improved stability and accuracy.

**Reference:** Penttilä, A., Shevchenko, V. G., Wilkman, O., & Muinonen, K. (2016). "Asteroid H, G1, G2 and H, G12 phase function performance with sparse data." Planetary and Space Science, 123, 117-122.

### Linear (H, β) Model

This is the simplest model, representing a first-order approximation of the phase curve as a straight line. It fits the absolute magnitude (H) at 0° phase angle and a linear phase coefficient (β, in mag/deg).

**Best Used For:** Datasets limited to small phase angles (typically α < 15°) where the opposition effect is not prominent. It is often used for quick characterization of comets and Trojans.

## Quick Start

Here is a simple example of how to use `pySPAC` to fit a phase curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from pyspac import PhaseCurve

# 1. Define observational data
phase_angles = np.array([5.2, 8.1, 12.5, 15.0, 18.9, 22.3])
magnitudes = np.array([15.45, 15.61, 15.89, 16.05, 16.35, 16.60])

# 2. Create a PhaseCurve object
phase_curve = PhaseCurve(angle=phase_angles, magnitude=magnitudes)

# 3. Fit a model (e.g., 'HG12')
try:
    fit_result = phase_curve.fitModel(model="HG12", method="trust-constr")
    print(fit_result.fit_report())

    # 4. Generate the fitted curve for plotting
    model_angles = np.linspace(0, 25, 100)
    model_mags = phase_curve.generateModel(model="HG12", degrees=model_angles)

    # 5. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(phase_angles, magnitudes, 'o', label='Observational Data')
    plt.plot(model_angles, model_mags, '-', label='Fitted HG12 Model')
    plt.gca().invert_yaxis()
    plt.xlabel("Phase Angle (degrees)")
    plt.ylabel("Reduced Magnitude")
    plt.title("Fitted Asteroid Phase Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"An error occurred during fitting: {e}")
```

## Contributing

Contributions are welcome! If you would like to contribute to the development of `pySPAC`, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear message.
4. Submit a pull request to the main branch of the original repository.

Please also feel free to open an issue to report bugs or suggest new features.

## Testing

To ensure the reliability of the photometric model implementations, `pySPAC` includes a test suite. To run the tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover tests
```

## Citations & Acknowledgements

If you use `pySPAC` in your research, please cite the foundational papers for the specific models you employ and this repository.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
