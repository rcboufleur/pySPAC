# Setting Up Objects

## PhaseCurve Constructor

```python
PhaseCurve(angle, magnitude=None, magnitude_unc=None, params=None,
           H=None, G=None, G12=None, G1=None, G2=None, beta=None)
```

## From Observational Data

### Basic Setup

```python
import numpy as np
from pyspac import PhaseCurve

# Required: angles and magnitudes
angles = np.array([0.17, 0.63, 0.98, 1.62, 4.95, 9.78,
                  12.94, 13.27, 13.81, 17.16, 18.52, 19.4])
mags = np.array([6.911, 7.014, 7.052, 7.105, 7.235, 7.341,
                7.425, 7.427, 7.437, 7.511, 7.551, 7.599])
pc = PhaseCurve(angle=angles, magnitude=mags)
```

### With Uncertainties (Recommended)

```python
# Include magnitude uncertainties for weighted fitting
angles = np.array([0.17, 0.63, 0.98, 1.62, 4.95, 9.78,
                  12.94, 13.27, 13.81, 17.16, 18.52, 19.4])
mags = np.array([6.911, 7.014, 7.052, 7.105, 7.235, 7.341,
                7.425, 7.427, 7.437, 7.511, 7.551, 7.599])
errors = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
                   0.02, 0.02, 0.03, 0.03, 0.04, 0.04])

pc = PhaseCurve(angle=angles, magnitude=mags, magnitude_unc=errors)
```

## From Model Parameters

### Individual Parameters

```python
# HG model
angles = np.linspace(0, 30, 100)
pc = PhaseCurve(angle=angles, H=15.0, G=0.25)

# HG1G2 model
pc = PhaseCurve(angle=angles, H=15.0, G1=0.3, G2=0.1)

# HG12 model
pc = PhaseCurve(angle=angles, H=15.0, G12=0.3)

# Linear model
pc = PhaseCurve(angle=angles, H=15.0, beta=0.04)
```

### Using Parameters Dictionary

```python
# HG parameters
hg_params = {"H": 15.0, "G": 0.25}
pc = PhaseCurve(angle=angles, params=hg_params)

# HG1G2 parameters
hg1g2_params = {"H": 15.0, "G1": 0.3, "G2": 0.1}
pc = PhaseCurve(angle=angles, params=hg1g2_params)
```

## Input Formats

```python
# Python lists
pc = PhaseCurve(angle=[5, 10, 15], magnitude=[15.2, 15.4, 15.7])

# Numpy arrays
angles = np.array([5, 10, 15])
mags = np.array([15.2, 15.4, 15.7])
pc = PhaseCurve(angle=angles, magnitude=mags)

# Single values
pc = PhaseCurve(angle=10.0, magnitude=15.5)
```

## Data Validation

```python
# Arrays must have same shape
try:
    pc = PhaseCurve(angle=[5, 10, 15], magnitude=[15.2, 15.4])  # Wrong!
except ValueError as e:
    print(f"Error: {e}")

# No NaN or infinite values allowed
try:
    pc = PhaseCurve(angle=[5, 10, np.nan], magnitude=[15.2, 15.4, 15.7])
except ValueError as e:
    print(f"Error: {e}")
```

## Object Properties

```python
pc = PhaseCurve(angle=[5, 10, 15], magnitude=[15.2, 15.4, 15.7])

# Data arrays (converted to numpy arrays internally)
print(pc.angle)        # [5 10 15]
print(pc.magnitude)    # [15.2 15.4 15.7]
print(pc.magnitude_unc)  # None if not provided

# Parameters dictionary
print(pc.params)       # {} initially, populated after fitting

# Fitting status
print(pc.fitting_status)  # False initially
print(pc.fitting_model)   # None initially
```

## Combining Data and Parameters

```python
# Start with parameters, then add observational data for comparison
pc = PhaseCurve(angle=angles, H=15.0, G=0.25)

# Later, you can still set observational data
pc.magnitude = observed_mags
pc.magnitude_unc = observed_errors
```

## String Representation

```python
pc = PhaseCurve(angle=[5, 10, 15], magnitude=[15.2, 15.4, 15.7])
print(pc)
# Output: <PhaseCurve object with 3 data points | Not yet fitted>
```

## Next Steps

- [Fitting Models](fitting-models.md) - Detailed fitting procedures
- [Setting Boundaries](boundaries.md) - Control parameter bounds and initial guesses
- [Models and Methods](models-methods.md) - Model/method compatibility details
