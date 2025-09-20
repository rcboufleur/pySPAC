# Setting Boundaries

## initial_conditions Parameter

The `initial_conditions` parameter in `fitModel()` controls parameter bounds and initial guesses.

```python
initial_conditions = [
    [initial_guess, min_boundary, max_boundary, vary_flag],
    [initial_guess, min_boundary, max_boundary, vary_flag],
    # ... one list per parameter
]
```

## Parameter Format

Each parameter list contains:
- **initial_guess** (float): Starting value for optimization
- **min_boundary** (float or None): Lower bound (None = no limit)
- **max_boundary** (float or None): Upper bound (None = no limit)
- **vary_flag** (bool): True to optimize, False to fix at initial_guess

## Parameter Order by Model

| Model | Parameter Order |
|-------|----------------|
| HG | [H_settings, G_settings] |
| HG12 | [H_settings, G12_settings] |
| HG12PEN | [H_settings, G12_settings] |
| HG1G2 | [H_settings, G1_settings, G2_settings] |
| LINEAR | [H_settings, beta_settings] |

## Basic Examples

### Setting Initial Guesses

```python
# HG model with custom starting values
initial_conditions = [
    [5.5, None, None, True],  # H: start at 5.5, no bounds, vary
    [0.20, None, None, True]   # G: start at 0.20, no bounds, vary
]

pc.fitModel(model="HG", method="trust-constr", initial_conditions=initial_conditions)
```

### Setting Boundaries

```python
# HG model with physical bounds
initial_conditions = [
    [5.0, 5.0, 20.0, True],  # H: range 5-20 magnitudes
    [0.15, 0.0, 1.0, True]     # G: range 0-1 (physical constraint)
]

pc.fitModel(model="HG", method="trust-constr", initial_conditions=initial_conditions)
```

### Fixing Parameters

```python
# Fix G at standard value, only fit H
initial_conditions = [
    [5.0, 5.0, 20.0, True],   # H: vary between 5-20
    [0.15, None, None, False]   # G: fixed at 0.15
]

pc.fitModel(model="HG", method="trust-constr", initial_conditions=initial_conditions)
```

## Model-Specific Examples

### HG Model

```python
hg_conditions = [
    [15.0, 5.0, 25.0, True],    # H: 5-25 mag range
    [0.15, 0.0, 1.0, True]      # G: 0-1 physical range
]

pc.fitModel(model="HG", method="trust-constr", initial_conditions=hg_conditions)
```

### HG1G2 Model

```python
hg1g2_conditions = [
    [15.0, 5.0, 25.0, True],    # H: absolute magnitude
    [0.30, 0.0, 1.0, True],     # G1: first slope parameter
    [0.15, 0.0, 1.0, True]      # G2: second slope parameter
]

pc.fitModel(model="HG1G2", method="trust-constr", initial_conditions=hg1g2_conditions)
```

### HG12/HG12PEN Models

```python
hg12_conditions = [
    [15.0, 5.0, 25.0, True],    # H: absolute magnitude
    [0.30, -0.1, 1.0, True]     # G12: can be slightly negative
]

pc.fitModel(model="HG12", method="trust-constr", initial_conditions=hg12_conditions)
pc.fitModel(model="HG12PEN", method="trust-constr", initial_conditions=hg12_conditions)
```

### LINEAR Model

```python
linear_conditions = [
    [15.0, 5.0, 25.0, True],    # H: absolute magnitude
    [0.04, 0.0, 0.1, True]      # beta: 0.01-0.08 mag/deg typical
]

pc.fitModel(model="LINEAR", method="leastsq", initial_conditions=linear_conditions)
```

## Multiple Starting Points

```python
# Try multiple starting points to avoid local minima
starting_points = [
    [[15.0, 5.0, 20.0, True], [0.15, 0.0, 1.0, True]],
    [[16.0, 5.0, 20.0, True], [0.25, 0.0, 1.0, True]],
    [[14.0, 5.0, 20.0, True], [0.10, 0.0, 1.0, True]]
]

best_rms = float('inf')
best_params = None

for conditions in starting_points:
    try:
        pc.fitModel(model="HG", method="trust-constr", initial_conditions=conditions)
        rms = np.sqrt(np.mean(np.array(pc.fit_residual)**2))

        if rms < best_rms:
            best_rms = rms
            best_params = pc.params.copy()
    except:
        continue

print(f"Best RMS: {best_rms:.4f}")
print(f"Best parameters: {best_params}")
```

## Data-Driven Initial Conditions

```python
# Estimate H from data
h_estimate = np.mean(mags)  # Rough estimate

# Estimate slope from data
if len(angles) >= 2:
    slope = (mags[-1] - mags[0]) / (angles[-1] - angles[0])
    g_estimate = np.clip(slope / 0.04, 0.0, 1.0)  # Convert to G
else:
    g_estimate = 0.15

# Use estimates as starting points
data_driven_conditions = [
    [h_estimate, h_estimate - 2, h_estimate + 2, True],
    [g_estimate, 0.0, 1.0, True]
]

pc.fitModel(model="HG", method="trust-constr", initial_conditions=data_driven_conditions)
```

## Next Steps

- [Models and Methods](models-methods.md) - Model/method compatibility details
- [Monte Carlo Uncertainties](uncertainties.md) - Parameter error estimation
- [Save and Load](save-load.md) - Save and load analysis results
