# Save and Load

## Overview

pySPAC provides JSON serialization to save and load complete analysis results.

- **`toJSON()`** - Serialize PhaseCurve object to JSON string
- **`fromJSON()`** - Reconstruct PhaseCurve object from JSON string

## Basic Save and Load

### Saving Results

```python
# Complete analysis
pc = PhaseCurve(angle=angles, magnitude=mags, magnitude_unc=errors)
pc.fitModel(model="HG", method="trust-constr")
pc.monteCarloUncertainty(n_simulations=5000, model="HG", method="trust-constr")

# Convert to JSON string
json_data = pc.toJSON()

# Save to file
with open('asteroid_analysis.json', 'w') as f:
    f.write(json_data)
```

### Loading Results

```python
# Load from file
with open('asteroid_analysis.json', 'r') as f:
    json_data = f.read()

# Reconstruct PhaseCurve object
pc_loaded = PhaseCurve.fromJSON(json_data)

# All data and results are preserved
pc_loaded.summary()
```

## What Gets Saved

### Complete State Preservation

The JSON format preserves:

- **Observational data**: `angle`, `magnitude`, `magnitude_unc`
- **Model parameters**: `params` dictionary
- **Fit results**: `fitting_status`, `fitting_model`, `fitting_method`, `fit_residual`
- **Monte Carlo data**: `montecarlo_uncertainty`, `uncertainty_results`, `uncertainty_source`
- **All other attributes**: Complete object state

### What's Not Saved

```python
# The lmfit ModelResult object is not saved (not JSON serializable)
# This means lmfit covariance errors are not available after reload
# Use Monte Carlo uncertainties for reloaded objects
```

## File Operations

### Simple File Save/Load

```python
def save_analysis(pc, filename):
    """Save PhaseCurve analysis to file."""
    json_data = pc.toJSON()
    with open(filename, 'w') as f:
        f.write(json_data)
    print(f"Saved to {filename}")

def load_analysis(filename):
    """Load PhaseCurve analysis from file."""
    with open(filename, 'r') as f:
        json_data = f.read()
    pc = PhaseCurve.fromJSON(json_data)
    print(f"Loaded from {filename}")
    return pc

# Usage
save_analysis(pc, 'my_asteroid.json')
pc_reloaded = load_analysis('my_asteroid.json')
```

### Error Handling

```python
def safe_save(pc, filename):
    """Save with error handling."""
    try:
        json_data = pc.toJSON()
        with open(filename, 'w') as f:
            f.write(json_data)
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False

def safe_load(filename):
    """Load with error handling."""
    try:
        with open(filename, 'r') as f:
            json_data = f.read()
        pc = PhaseCurve.fromJSON(json_data)
        return pc
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filename}")
    except Exception as e:
        print(f"Load failed: {e}")
    return None

# Usage
if safe_save(pc, 'analysis.json'):
    pc_loaded = safe_load('analysis.json')
```

## Batch Operations

### Save Multiple Objects

```python
def save_batch_analyses(objects_dict, filename):
    """Save multiple PhaseCurve objects to one file."""

    batch_data = {}
    for name, pc in objects_dict.items():
        try:
            batch_data[name] = pc.toJSON()
        except Exception as e:
            print(f"Failed to save {name}: {e}")

    with open(filename, 'w') as f:
        json.dump(batch_data, f, indent=2)

    print(f"Saved {len(batch_data)} objects to {filename}")

def load_batch_analyses(filename):
    """Load multiple PhaseCurve objects from one file."""

    with open(filename, 'r') as f:
        batch_data = json.load(f)

    objects = {}
    for name, json_str in batch_data.items():
        try:
            objects[name] = PhaseCurve.fromJSON(json_str)
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    print(f"Loaded {len(objects)} objects from {filename}")
    return objects

# Usage
asteroids = {
    'Ceres': pc1,
    'Vesta': pc2,
    'Pallas': pc3
}

save_batch_analyses(asteroids, 'asteroid_batch.json')
loaded_asteroids = load_batch_analyses('asteroid_batch.json')
```

### Process Multiple Files

```python
import glob
import os

def process_directory(input_dir, output_dir):
    """Process all JSON files in a directory."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    for filepath in json_files:
        try:
            # Load object
            pc = safe_load(filepath)
            if pc is None:
                continue

            # Process (example: add Monte Carlo if missing)
            if pc.uncertainty_results is None and pc.fitting_status:
                pc.monteCarloUncertainty(
                    n_simulations=3000,
                    model=pc.fitting_model,
                    method="trust-constr",
                    verbose=False
                )

            # Save processed version
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            safe_save(pc, output_path)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

# Usage
process_directory('raw_analyses/', 'processed_analyses/')
```

## Data Management

### Add Metadata

```python
def save_with_metadata(pc, filename, metadata=None):
    """Save with additional metadata."""

    # Get base object data
    json_str = pc.toJSON()
    base_data = json.loads(json_str)

    # Add metadata wrapper
    save_data = {
        'pySPAC_version': '1.0',
        'save_date': '2024-01-15',
        'metadata': metadata or {},
        'phasecurve_data': base_data
    }

    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)

def load_with_metadata(filename):
    """Load and extract metadata."""

    with open(filename, 'r') as f:
        save_data = json.load(f)

    # Extract metadata
    metadata = {
        'version': save_data.get('pySPAC_version'),
        'save_date': save_data.get('save_date'),
        'custom': save_data.get('metadata', {})
    }

    # Reconstruct object
    if 'phasecurve_data' in save_data:
        pc = PhaseCurve.fromJSON(json.dumps(save_data['phasecurve_data']))
    else:
        # Fallback for simple format
        pc = PhaseCurve.fromJSON(json.dumps(save_data))
        metadata = {}

    return pc, metadata

# Usage
metadata = {
    'object_name': '433 Eros',
    'observer': 'Smith et al.',
    'telescope': 'Palomar 200-inch',
    'filter': 'V-band'
}

save_with_metadata(pc, 'eros_analysis.json', metadata)
pc_loaded, meta = load_with_metadata('eros_analysis.json')
print(f"Metadata: {meta}")
```

## Compressed Storage

```python
import gzip

def save_compressed(pc, filename):
    """Save with gzip compression."""

    json_str = pc.toJSON()

    with gzip.open(filename, 'wt') as f:
        f.write(json_str)

    # Report compression ratio
    original_size = len(json_str)
    compressed_size = os.path.getsize(filename)
    ratio = compressed_size / original_size

    print(f"Compressed: {original_size:,} → {compressed_size:,} bytes ({ratio:.1%})")

def load_compressed(filename):
    """Load compressed file."""

    with gzip.open(filename, 'rt') as f:
        json_str = f.read()

    return PhaseCurve.fromJSON(json_str)

# Usage
save_compressed(pc, 'analysis.json.gz')
pc_loaded = load_compressed('analysis.json.gz')
```

## Validation After Loading

```python
def validate_loaded_object(pc):
    """Validate loaded PhaseCurve object."""

    issues = []

    # Check basic data
    if pc.angle is None or pc.magnitude is None:
        issues.append("Missing observational data")
    elif len(pc.angle) != len(pc.magnitude):
        issues.append("Data arrays have different lengths")

    # Check fit consistency
    if pc.fitting_status and not pc.params:
        issues.append("Fitting status true but no parameters")

    # Check Monte Carlo consistency
    if pc.montecarlo_uncertainty and not pc.uncertainty_results:
        issues.append("Monte Carlo samples exist but no processed results")

    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Object validation passed")
        return True

# Usage
pc_loaded = safe_load('analysis.json')
if pc_loaded and validate_loaded_object(pc_loaded):
    pc_loaded.summary()
```

## Next Steps

- [Plotting Results](plotting.md) - Plot and display results
- [Generate Models](generate-models.md) - Model generation from known parameters
