# Fitting Models

## fitModel Method

```python
fitModel(model, method, initial_conditions=None)
```

Fits a photometric model to observational data using weighted least-squares.

## Basic Fitting

```python
# Create object with data
pc = PhaseCurve(angle=angles, magnitude=mags, magnitude_unc=errors)

# Fit HG model
pc.fitModel(model="HG", method="trust-constr")

# View results
pc.summary()
```
```console
================== PhaseCurve Summary ==================
Data Points:         12
Angle Range:         0.17° to 19.40°
Magnitude Range:     6.91 to 7.60
------------------------------------------------------
Fitting Status:      SUCCESS
Model:               HG
Method:              trust-constr
Fit RMS:             0.9365
------------------- Model Parameters -------------------
  H                    6.9365
  G                    0.4731
======================================================
```

## Available Models

### HG Model

```python
pc.fitModel(model="HG", method="trust-constr")
print(f"H = {pc.params['H']:.3f}")
print(f"G = {pc.params['G']:.3f}")
```

- **Parameters**: H (absolute magnitude), G (slope parameter)
- **Reference**: Bowell et al. (1989)

### HG1G2 Model

```python
pc.fitModel(model="HG1G2", method="trust-constr")
print(f"H = {pc.params['H']:.3f}")
print(f"G1 = {pc.params['G1']:.3f}")
print(f"G2 = {pc.params['G2']:.3f}")
```

- **Parameters**: H, G1 (first slope), G2 (second slope). Physically-based three-parameter system.
- **Reference**: Muinonen et al. (2010)

### HG12 Model

```python
pc.fitModel(model="HG12", method="trust-constr")
print(f"H = {pc.params['H']:.3f}")
print(f"G12 = {pc.params['G12']:.3f}")
print(f"G1 = {pc.params['G1']:.3f} (derived)")
print(f"G2 = {pc.params['G2']:.3f} (derived)")
```

- **Parameters**: H, G12 (combined slope parameter). Simplified two-parameter version of HG1G2.
- **Derived**: G1 and G2 calculated from G12
- **Reference**: Muinonen et al. (2010)

### HG12PEN Model

```python
pc.fitModel(model="HG12PEN", method="trust-constr")
```

- **Parameters**: Same as HG12 but improved calibration. Recalibrated HG12 for better sparse data performance.
- **Reference**: Penttilä et al. (2016)

### LINEAR Model

```python
pc.fitModel(model="LINEAR", method="leastsq")
print(f"H = {pc.params['H']:.3f}")
print(f"beta = {pc.params['beta']:.4f} mag/deg")
```

- **Parameters**: H, β (linear phase coefficient in mag/deg). First-order approximation for small phase angles.

## Fit Results

```python
# After fitting, the object is updated
print(f"Fitting status: {pc.fitting_status}")      # True
print(f"Model: {pc.fitting_model}")                # "HG"
print(f"Method: {pc.fitting_method}")              # "trust-constr"
print(f"Residuals: {len(pc.fit_residual)} points") # List of residuals
```

## Error Handling

```python
try:
    pc.fitModel(model="HG", method="trust-constr")
except RuntimeError as e:
    print(f"Fitting failed: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

## Multiple Fits

```python
# Try different models
models = ["HG", "HG12", "LINEAR"]
results = {}

for model in models:
    try:
        method = "trust-constr" if model != "LINEAR" else "leastsq"
        pc.fitModel(model=model, method=method)
        rms = np.sqrt(np.mean(np.array(pc.fit_residual)**2))
        results[model] = {'rms': rms, 'params': pc.params.copy()}
    except Exception as e:
        results[model] = {'error': str(e)}

# Find best fit
best_model = min([k for k in results if 'rms' in results[k]],
                key=lambda x: results[x]['rms'])
print(f"Best model: {best_model}")
```

## Weighted vs Unweighted Fitting

```python
# Weighted fitting (when magnitude_unc provided)
pc = PhaseCurve(angle=angles, magnitude=mags, magnitude_unc=errors)
pc.fitModel(model="HG", method="trust-constr")  # Uses weights

# Unweighted fitting
pc = PhaseCurve(angle=angles, magnitude=mags)  # No magnitude_unc
pc.fitModel(model="HG", method="trust-constr")  # Equal weights
```

## Next Steps

- [Setting Boundaries](boundaries.md) - Control parameter bounds and initial guesses
- [Models and Methods](models-methods.md) - Model/method compatibility details
- [Monte Carlo Uncertainties](uncertainties.md) - Parameter error estimation
