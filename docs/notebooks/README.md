# Notebook Guidelines

This directory contains Jupyter notebooks that demonstrate jaxdf functionality. These notebooks are automatically tested to ensure they remain functional as the library evolves.

## Creating Notebooks

### Basic Requirements

1. **Execute all cells** before committing (notebooks should have outputs)
2. **Use descriptive names**: `feature_name.ipynb` not `notebook1.ipynb`
3. **Keep execution time reasonable**: Aim for <2 minutes total
4. **Use CPU-friendly code**: Tests run with `JAX_PLATFORMS=cpu`

### Making Notebooks Testable

#### Use Fixed Random Seeds

For reproducible outputs:
```python
import jax
import jax.random as jr

# Good - fixed seed
key = jr.PRNGKey(42)

# Avoid - non-deterministic
key = jr.PRNGKey()  # Uses system time
```

#### Mark Non-Deterministic Cells

For cells with varying outputs (plots, random numbers without seeds), add:

```python
# SKIP_COMPARE
# This cell's output varies, skip in regression tests

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
```

Or use cell metadata (View → Cell Toolbar → Tags → Add `nbval-skip`)

#### Patterns That Auto-Skip

These patterns are automatically ignored in output comparison:
- `plt.show()` and `fig.show()`
- Timestamps and dates
- Memory addresses
- JAX platform warnings

#### Avoid Time-Dependent Code

```python
# Avoid
import time
start = time.time()  # Output will differ

# Better
# Just show the algorithm, don't time it in the notebook
```

### Structure Guidelines

#### Start with Setup

```python
# Cell 1: Imports and configuration
import jax
import jax.numpy as jnp
from jaxdf import FourierSeries, Domain

# Force CPU for consistency
import os
os.environ["JAX_PLATFORMS"] = "cpu"
```

#### Include Documentation

```markdown
# Notebook Title

This notebook demonstrates how to use jaxdf for [purpose].

## Contents
1. Setup
2. Basic usage
3. Advanced examples
```

#### End with Summary

```markdown
## Summary

In this notebook we showed:
- Feature X
- Feature Y
- How to combine them
```

### Example Cell Patterns

#### Good: Deterministic Output
```python
# Cell with reproducible output
domain = Domain((64, 64), (1.0, 1.0))
params = jnp.ones((64, 64, 1))
field = FourierSeries(params, domain)

print(f"Field shape: {field.params.shape}")
print(f"Domain size: {field.domain.size}")
```

#### Good: Non-Deterministic with Marker
```python
# SKIP_COMPARE
# Visualization - output may vary

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(field.on_grid[..., 0])
plt.show()
```

#### Avoid: Hidden Non-Determinism
```python
# Bad - uses random seed from system time internally
# Will fail output comparison tests unpredictably
model = SomeModel()  # Internally uses np.random without seed
```

## Testing Notebooks

### Local Testing

```bash
# Test execution only
./scripts/test_notebooks.sh your_notebook.ipynb

# Test with output comparison
pytest tests/test_notebooks.py --nbval -k your_notebook
```

### Updating Outputs

After intentional changes to library code:

```bash
# Update specific notebook
./scripts/update_notebook_outputs.sh your_notebook.ipynb

# Update all notebooks
./scripts/update_notebook_outputs.sh
```

### CI Testing

Notebooks are automatically tested in CI on:
- Pull requests
- Weekly schedule
- Manual trigger

See `docs/NOTEBOOK_TESTING.md` for full testing documentation.

## Slow Notebooks

If your notebook takes >60 seconds to execute:

1. **Optimize if possible**: Use smaller datasets, fewer iterations
2. **Mark as slow** in `tests/test_notebooks.py`:
   ```python
   SLOW_NOTEBOOKS = {
       "expensive_simulation.ipynb",
   }
   ```
3. **Consider splitting**: Create separate notebooks for quick examples vs. full demos