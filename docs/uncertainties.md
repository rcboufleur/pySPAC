# Monte Carlo Uncertainties

## Overview

pySPAC provides two Monte Carlo methods for parameter uncertainty estimation:

1. **`monteCarloUncertainty()`** - Uses only observational errors
2. **`monteCarloUnknownRotation()`** - Estimates rotational lightcurve scatter

## Method 1: Observational Uncertainties

### Basic Usage

```python
# Fit model first
pc.fitModel(model="HG", method="trust-constr")

# Run Monte Carlo with observational errors
pc.monteCarloUncertainty(
    n_simulations=500,
    model="HG",
    method="trust-constr"
)

# View results
pc.summary()
```

### Method Signature

```python
monteCarloUncertainty(n_simulations, model, method, n_threads=1, verbose=True)
```

**Parameters:**

- **n_simulations** (int): Number of Monte Carlo iterations
- **model** (str): Model to fit in each simulation
- **method** (str): Optimization method
- **n_threads** (int): Number of parallel threads (default: 1)
- **verbose** (bool): Show progress bar (default: True)

### How It Works

1. Generate simulated datasets by adding Gaussian noise based on `magnitude_unc`
2. Fit the model to each noisy dataset
3. Collect successful parameter estimates
4. Calculate percentile-based uncertainties

### Without Magnitude Uncertainties

```python
# If no uncertainties provided, pySPAC estimates from RMS
pc = PhaseCurve(angle=angles, magnitude=mags)  # No magnitude_unc
pc.fitModel(model="HG", method="trust-constr")

# Automatically estimates uniform uncertainties
pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")
```

## Method 2: Unknown Rotational Variation

### Basic Usage

```python
# Include rotational scatter in uncertainty analysis
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=0.15,  # 0.15 mag rotational amplitude
    model="HG",
    distribution='sinusoidal',
    method="trust-constr",
)
```

### Method Signature

```python
monteCarloUnknownRotation(n_simulations, amplitude_variation, model,
                         distribution="sinusoidal", method="trust-constr",
                         n_threads=1, verbose=True)
```

**Parameters:**

- **amplitude_variation** (float): Semi-amplitude of rotational variation (magnitudes)
- **distribution** (str): "sinusoidal" or "uniform" (default: "sinusoidal")
- Other parameters same as `monteCarloUncertainty`

### Rotational Amplitude Guide

```python
# Typical asteroid rotational amplitudes
amplitudes = {
    'spherical': 0.05,      # Nearly spherical objects
    'typical': 0.15,        # Average asteroids
    'elongated': 0.25,      # Highly elongated objects
    'binary': 0.40,         # Binary systems
    'extreme': 0.60         # Extreme cases
}

# Usage
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=amplitudes['typical'],
    model="HG",
    distribution='sinusoidal',
    method="trust-constr"
)
```

### Distribution Types

```python
# Sinusoidal distribution (realistic for most asteroids)
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=0.15,
    distribution="sinusoidal",
    model="HG",
    method="trust-constr"
)

# Uniform distribution (conservative estimate)
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=0.15,
    distribution="uniform",
    model="HG",
    method="trust-constr"
)
```

## Using Different Percentiles

### Default Percentiles (68% confidence)

```python
# Default: 15.87%, 50%, 84.13% (±1σ equivalent)
pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")
pc.summary()  # Shows default 68% confidence intervals
```

### Custom Confidence Levels

```python
# 95% confidence interval (2σ equivalent)
pc.calculate_uncertainties(percentiles=[2.5, 50, 97.5])
pc.summary()

# 99% confidence interval (3σ equivalent)
pc.calculate_uncertainties(percentiles=[0.5, 50, 99.5])
pc.summary()

# 90% confidence interval
pc.calculate_uncertainties(percentiles=[5, 50, 95])
pc.summary()
```

### Multiple Confidence Levels

```python
def show_multiple_confidence_levels(pc):
    """Display uncertainties at multiple confidence levels."""

    levels = [
        ([15.87, 50, 84.13], "68% (1σ)"),
        ([2.5, 50, 97.5], "95% (2σ)"),
        ([0.5, 50, 99.5], "99% (3σ)")
    ]

    print(f"{'Parameter':<10} {'Level':<10} {'Median':<8} {'Lower':<8} {'Upper':<8}")
    print("-" * 50)

    for percentiles, label in levels:
        pc.calculate_uncertainties(percentiles=percentiles)

        for param, stats in pc.uncertainty_results.items():
            median = stats['median']
            lower = stats['lower_error']
            upper = stats['upper_error']
            print(f"{param:<10} {label:<10} {median:<8.4f} {lower:<8.4f} {upper:<8.4f}")

# Usage after Monte Carlo analysis
show_multiple_confidence_levels(pc)
```

## Accessing Results

### Raw Monte Carlo Samples

```python
# Access raw parameter samples
mc_samples = pc.montecarlo_uncertainty

for param, values in mc_samples.items():
    print(f"{param}: {len(values)} samples")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Std: {np.std(values):.4f}")
```

### Processed Uncertainties

```python
# Access processed uncertainty statistics
uncertainties = pc.uncertainty_results

for param, stats in uncertainties.items():
    print(f"{param}:")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Upper error: +{stats['upper_error']:.4f}")
    print(f"  Lower error: {stats['lower_error']:.4f}")
    print(f"  Percentiles: {stats['percentiles']}")
```


## Thread Usage

```python
import multiprocessing as mp

# Use multiple threads for faster computation
n_threads = min(4, mp.cpu_count())  # Don't exceed 4 threads

pc.monteCarloUncertainty(
    n_simulations=500,
    model="HG",
    method="trust-constr",
    n_threads=n_threads
)
```

## Choosing Between Methods

### Decision Framework

```python
def choose_uncertainty_method(has_uncertainties, rotational_knowledge):
    """Choose appropriate uncertainty method."""

    if rotational_knowledge == "negligible":
        # Rotational amplitude < 0.05 mag
        return "monteCarloUncertainty"

    elif rotational_knowledge == "unknown":
        # Unknown rotational state - conservative approach
        return "monteCarloUnknownRotation"

    elif rotational_knowledge == "significant":
        # Known large rotational amplitude
        return "monteCarloUnknownRotation"

    elif not has_uncertainties:
        # No observational errors - estimate both sources
        return "monteCarloUnknownRotation"

    else:
        # Standard case with observational errors
        return "monteCarloUncertainty"

# Usage
method = choose_uncertainty_method(
    has_uncertainties=True,
    rotational_knowledge="unknown"
)
print(f"Recommended method: {method}")
```

### Typical Use Cases

```python
# Case 1: High-quality photometry, known spherical object
pc.monteCarloUncertainty(n_simulations=500, model="HG", method="trust-constr")

# Case 2: Standard photometry, unknown rotation
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=0.15,  # Typical asteroid
    model="HG",
    method="trust-constr"
)

# Case 3: Sparse data, possibly elongated object
pc.monteCarloUnknownRotation(
    n_simulations=500,
    amplitude_variation=0.25,  # Conservative estimate
    model="HG",
    distribution="uniform",    # Conservative distribution
    method="trust-constr"
)
```

## Validation and Diagnostics

### Convergence Check

```python
def check_convergence(pc, target_precision=0.02):
    """Check if Monte Carlo has converged."""

    samples = pc.montecarlo_uncertainty

    for param, values in samples.items():
        n_samples = len(values)

        # Split into two halves
        half = n_samples // 2
        first_half = values[:half]
        second_half = values[half:2*half]

        # Compare standard deviations
        std1 = np.std(first_half)
        std2 = np.std(second_half)

        relative_diff = abs(std1 - std2) / max(std1, std2)

        if relative_diff > target_precision:
            print(f"WARNING: {param} may not be converged ({relative_diff:.4f})")
        else:
            print(f"OK: {param} converged ({relative_diff:.4f})")

# Usage after Monte Carlo
check_convergence(pc)
```

### Sample Quality

```python
def analyze_sample_quality(pc):
    """Analyze quality of Monte Carlo samples."""

    samples = pc.montecarlo_uncertainty

    for param, values in samples.items():
        # Basic statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        # Skewness and kurtosis
        from scipy import stats
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)

        print(f"{param}:")
        print(f"  Samples: {len(values)}")
        print(f"  Mean: {mean_val:.4f}, Median: {median_val:.4f}")
        print(f"  Std: {std_val:.4f}")
        print(f"  Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")

        # Flag potential issues
        if abs(skewness) > 1:
            print(f"  WARNING: High skewness")
        if abs(kurtosis) > 3:
            print(f"  WARNING: High kurtosis")

# Usage
analyze_sample_quality(pc)
```

## Performance Optimization

### Efficient Settings

```python
# For production analysis
pc.monteCarloUncertainty(
    n_simulations=500,
    model="HG",
    method="trust-constr",
    n_threads=4,
    verbose=False  # Disable progress bar
)

# For interactive exploration
pc.monteCarloUncertainty(
    n_simulations=100,  # Fewer simulations for speed
    model="HG",
    method="trust-constr",
    verbose=True
)
```
