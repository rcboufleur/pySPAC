# pySPAC: Solar Phase curve Analysis and Characterization

pySPAC (python Solar Phase curve Analysis and Characterization) is a library for analyzing and fitting astronomical phase curves of asteroids and other small solar system bodies. It provides an object-oriented interface to bridge observational photometric data with standard theoretical models. The library is built on the numerical frameworks of lmfit, sbpy, and numpy.

The primary function of pySPAC is to determine the absolute magnitude (H) and other phase function parameters from a set of phase angle (α) and reduced magnitude observations.

## Features

- Provides an object-oriented interface via the PhaseCurve class.
- Supports standard IAU photometric models: (H, G), (H, G₁, G₂), (H, G₁₂), and a linear fit.
- Performs weighted least-squares fitting when observational uncertainties are provided.
- Includes two Monte Carlo methods for uncertainty analysis: one based on observational errors and another that includes rotational scatter.
- Estimates uniform uncertainty from the RMS of residuals if observational errors are not provided.
- Allows selection from various lmfit optimization methods with built-in validation based on model constraints.
- Generates a summary report of fit results and calculated uncertainties.
- Saves and loads the object state to and from JSON.

## Installation

### From PyPI
To get the latest stable release, use pip:

```bash
pip install pyspac
```

### From GitHub
To install the latest development version directly from the source code:

```bash
pip install git+https://github.com/rcboufleur/pySPAC.git
```

## Usage Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pyspac import PhaseCurve

# 1. Define observational data, including uncertainties
phase_angles = np.array([5.2, 8.1, 12.5, 15.0, 18.9, 22.3])
magnitudes = np.array([15.45, 15.61, 15.89, 16.05, 16.35, 16.60])
mag_unc = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.04])

# 2. Create a PhaseCurve object
pc = PhaseCurve(angle=phase_angles, magnitude=magnitudes, magnitude_unc=mag_unc)

# 3. Fit a model and print the initial summary
try:
    pc.fitModel(model="HG", method="leastsq")
    print("--- Summary After Initial Fit ---")
    pc.summary()

    # 4. Run a Monte Carlo simulation for detailed uncertainties
    print("\nRunning Monte Carlo simulation...")
    pc.monteCarloUncertainty(n_simulations=5000, model="HG", method="leastsq")

    # 5. View the final summary with MC uncertainties
    print("\n--- Final Summary with Monte Carlo Uncertainties ---")
    pc.summary()

    # 6. Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(phase_angles, magnitudes, yerr=mag_unc, fmt='o', capsize=5, label='Observational Data')
    model_angles = np.linspace(0, 25, 100)
    model_mags = pc.generateModel(model="HG", degrees=model_angles)
    plt.plot(model_angles, model_mags, '-', label='Fitted HG Model')
    plt.gca().invert_yaxis()
    plt.xlabel("Phase Angle (degrees)")
    plt.ylabel("Reduced Magnitude")
    plt.title("Fitted Asteroid Phase Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

except (RuntimeError, ValueError) as e:
    print(f"An error occurred: {e}")
```

## Additional Features

### Uncertainty Analysis
pySPAC offers two Monte Carlo methods for different scientific cases:

- `.monteCarloUncertainty()`: Estimates parameter uncertainties based solely on the provided `magnitude_unc`.
- `.monteCarloUnknownRotation()`: Models both the observational error and an additional source of scatter from the asteroid's unknown rotation, controlled by the `amplitude_variation` parameter.

### Saving and Loading State
The state of a PhaseCurve object can be saved to and reloaded from a file.

```python
# Save the fitted object's data to a JSON string
json_data = pc.toJSON()
with open('my_asteroid_fit.json', 'w') as f:
    f.write(json_data)

# Load the data back into a new object
with open('my_asteroid_fit.json', 'r') as f:
    json_data_from_file = f.read()
reloaded_pc = PhaseCurve.fromJSON(json_data_from_file)
reloaded_pc.summary()
```

## The Photometric Models

pySPAC supports several widely-used photometric systems. Choosing the correct model depends on the quality of your data and the physical properties of the object being studied.

### The IAU (H, G) System
The (H, G) system was the standard model adopted by the IAU. It describes the phase curve using the absolute magnitude (H) and a single slope parameter (G).

**Reference:** Bowell, E., et al. (1989). "Application of photometric models to asteroids." In Asteroids II (pp. 524-556). University of Arizona Press.

### The (H, G₁, G₂) System
A more physically-based model that better describes the shape of the phase curve and opposition effect. The G₁ and G₂ parameters are related to the surface properties of the body.

**Reference:** Muinonen, K., et al. (2010). "A three-parameter magnitude phase function for asteroids." Icarus, 209(2), 542-555.

### The (H, G₁₂) System
A simplified, two-parameter version of the (H, G₁, G₂) model, introduced for cases where data is too sparse to constrain all three parameters.

**Reference:** Muinonen, K., et al. (2010). "A three-parameter magnitude phase function for asteroids." Icarus, 209(2), 542-555.

### The (H, G₁₂) System - Penttilä et al. (2016) calibration (HG12PEN)
A recalibration of the (H, G₁₂) model to improve performance with sparse data.

**Reference:** Penttilä, A., et al. (2016). "Asteroid H, G1, G2 and H, G12 phase function performance with sparse data." Planetary and Space Science, 123, 117-122.

### Linear (H, β) Model
A first-order approximation of the phase curve as a straight line, fitting the absolute magnitude (H) and a linear phase coefficient (β, in mag/deg). Often used for datasets limited to small phase angles.

## Contributing

Contributions are welcome. If you would like to contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Submit a pull request to the main branch.

Please also feel free to open an issue to report bugs or suggest new features.

## Testing

To run the test suite, navigate to the root directory of the project and run:

```bash
python -m unittest discover tests
```

## Citations & Acknowledgements

If you use pySPAC in your research, please cite the foundational papers for the specific models you employ, and provide a link to this repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
