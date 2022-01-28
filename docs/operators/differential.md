# `jaxdf.operators.differential`

## Implementations

```python
differential.py:
----------------
derivative:
 ─ (x: Continuous, axis=0, params=None)
diag_jacobian:
 ─ (x: FourierSeries, params=None)
 ─ (x: Continuous, params=None)
gradient:
 ─ (x: FiniteDifferences, params=None, accuracy=4)
 ─ (x: FourierSeries, params=None)
 ─ (x: Continuous, params=None)
laplacian:
 ─ (x: FiniteDifferences, params=None, accuracy=4)
 ─ (x: FourierSeries, params=None)
 ─ (x: Continuous, params=None)
```