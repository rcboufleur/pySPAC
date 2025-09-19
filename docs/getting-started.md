# Getting Started

## Installation

```bash
pip install pyspac
```

## Basic Workflow

1. Create PhaseCurve object with observational data
2. Fit a photometric model
3. Estimate uncertainties with Monte Carlo
4. Save results

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pyspac import PhaseCurve

# Step 1: Define observational data
phase_angles = np.array([0.17, 0.63, 0.98, 1.62, 4.95, 9.78,
                        12.94, 13.27, 13.81, 17.16, 18.52, 19.4])
magnitudes = np.array([6.911, 7.014, 7.052, 7.105, 7.235, 7.341,
                      7.425, 7.427, 7.437, 7.511, 7.551, 7.599])
mag_unc = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
                   0.02, 0.02, 0.03, 0.03, 0.04, 0.04])

# Step 2: Create PhaseCurve object
pc = PhaseCurve(angle=phase_angles, magnitude=magnitudes, magnitude_unc=mag_unc)

# Step 3: Fit model
pc.fitModel(model="HG", method="trust-constr")
print("Initial fit:")
pc.summary()

# Step 4: Monte Carlo uncertainties
pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")
print("With uncertainties:")
pc.summary()

# Step 5: Plot results
model_angles = np.linspace(0, 25, 100)
model_mags = pc.generateModel(model="HG", degrees=model_angles)

plt.errorbar(phase_angles, magnitudes, yerr=mag_unc, fmt='o', label='Data')
plt.plot(model_angles, model_mags, '-', label='HG Model')
plt.gca().invert_yaxis()
plt.xlabel("Phase Angle (degrees)")
plt.ylabel("Reduced Magnitude")
plt.legend()
plt.show()

# Step 6: Save results
json_data = pc.toJSON()
with open('results.json', 'w') as f:
    f.write(json_data)
```

## Models and Data Requirements

| Model | Parameters | Min Points | Recommended Points |
|-------|------------|------------|-------------------|
| LINEAR | 2 | 3 | 5+ |
| HG | 2 | 3 | 5+ |
| HG12 | 2 | 3 | 5+ |
| HG12PEN | 2 | 3 | 5+ |
| HG1G2 | 3 | 3* | 5+ |

*Due to constraint 1-G1-G2=0

## Error Handling

```python
try:
    pc.fitModel(model="HG", method="trust-constr")
    pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")
except ValueError as e:
    print(f"Parameter error: {e}")
except RuntimeError as e:
    print(f"Fitting failed: {e}")
```

## Next Steps

- [Setting Up Objects](setup-objects.md) - Different ways to create PhaseCurve objects
- [Fitting Models](fitting-models.md) - Detailed fitting procedures
- [Monte Carlo Uncertainties](uncertainties.md) - Error estimation methods
