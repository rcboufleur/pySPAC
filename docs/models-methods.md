# Models and Methods

## Model-Method Compatibility

Different models have mathematical constraints that require specific optimization methods.

### Constrained Models

These models have mathematical constraints and **must** use constraint-capable methods:

| Model | Constraints | Required Methods |
|-------|-------------|------------------|
| HG | G constraints | SLSQP, COBYLA, trust-constr |
| HG12 | G12 constraints | SLSQP, COBYLA, trust-constr |
| HG12PEN | G12 constraints | SLSQP, COBYLA, trust-constr |
| HG1G2 | G1+G2 constraints | SLSQP, COBYLA, trust-constr |

### Unconstrained Models

This model can use any optimization method:

| Model | Constraints | Available Methods |
|-------|-------------|------------------|
| LINEAR | None (simple bounds) | All methods |

## Available Methods

### Constraint-Capable Methods

Required for HG, HG12, HG12PEN, HG1G2:

```python
# These methods handle mathematical constraints
pc.fitModel(model="HG", method="trust-constr")  # Most robust
pc.fitModel(model="HG", method="SLSQP")         # Fast, reliable
pc.fitModel(model="HG", method="COBYLA")        # Good for poor data
```

### All Methods

Available for LINEAR model:

```python
# Constraint-capable methods
methods = ['trust-constr', 'SLSQP', 'COBYLA']

# Additional methods for LINEAR only
methods += ['L-BFGS-B', 'TNC', 'Powell', 'least_squares',
           'BFGS', 'CG', 'Nelder-Mead', 'leastsq']

# Example
pc.fitModel(model="LINEAR", method="leastsq")  # Fast for unconstrained
```

## Method Characteristics

| Method | Speed | Robustness | Notes |
|--------|-------|------------|-------|
| trust-constr | Slow | Excellent | Most reliable for difficult cases |
| SLSQP | Medium | Good | Good balance of speed and reliability |
| COBYLA | Fast | Good | Tolerant of poor data quality |
| leastsq | Very Fast | Medium | LINEAR only, good for clean data |
| L-BFGS-B | Fast | Good | LINEAR only, handles bounds well |

## Error Handling

```python
# pySPAC automatically checks compatibility
try:
    pc.fitModel(model="HG", method="leastsq")  # Invalid!
except ValueError as e:
    print(e)
    # Output: Model 'HG' is constrained and must use a method from: ['SLSQP', 'COBYLA', 'trust-constr']
```

## Model Complexity and Data Requirements

### Parameter Count

| Model | Parameters | Min Points | Recommended Points |
|-------|------------|------------|-------------------|
| LINEAR | 2 (H, β) | 3 | 5+ |
| HG | 2 (H, G) | 3 | 5+ |
| HG12 | 2 (H, G12) | 3 | 5+ |
| HG12PEN | 2 (H, G12) | 3 | 5+ |
| HG1G2 | 3 (H, G1, G2) | 3* | 5+ |

*Due to constraint 1-G1-G2=0

## Next Steps

- [Monte Carlo Uncertainties](uncertainties.md) - Parameter error estimation
- [Save and Load](save-load.md) - Save and load analysis results
- [Plotting Results](plotting.md) - Plot and display results
