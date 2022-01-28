# `jaxdf.operators.magic`

🛠️ **Documentation is work in progress**

The functions in this module override the corresponding magic methods of the `Field` class and derived classes.

## Available Implementations

```python
magic.py:
----------------
__add__:
 ─ (x: OnGrid, y: object, params=None)
 ─ (x: Continuous, y: Continuous, params=None)
 ─ (x: Linear, y: Linear, params=None)
 ─ (x: Continuous, y: object, params=None)
__bool__:
 ─ (x: OnGrid, params=None)
__divmod__:
 ─ (x: OnGrid, y: OnGrid, params=None)
 ─ (x: Linear, y, params=None)
__float__:
 ─ (x: OnGrid, params=None)
__mul__:
 ─ (x: Continuous, y, params=None)
 ─ (x: Linear, y, params=None)
 ─ (x: Continuous, y: Continuous, params=None)
 ─ (x: OnGrid, y: OnGrid, params=None)
__neg__:
 ─ (x: Linear, params=None)
 ─ (x: Continuous, params=None)
__pow__:
 ─ (x: OnGrid, y: object, params=None)
 ─ (x: Continuous, y: Continuous, params=None)
 ─ (x: OnGrid, y: OnGrid, params=None)
 ─ (x: Continuous, y: object, params=None)
__radd__:
 ─ (x: OnGrid, y: object, params=None)
 ─ (x: Continuous, y: object, params=None)
__rmul__:
 ─ (x: jaxdf.core.Field, y: object, params=None)
__rpow__:
 ─ (x: OnGrid, y: object, params=None)
__rsub__:
 ─ (x: Linear, y: object, params=None)
__rtruediv__:
 ─ (x: OnGrid, y: object, params=None)
 ─ (x: Continuous, y: object, params=None)
__sub__:
 ─ (x: OnGrid, y: object, params=None)
 ─ (x: Linear, y: Linear, params=None)
__truediv__:
 ─ (x: Linear, y, params=None)
 ─ (x: Continuous, y: Continuous, params=None)
 ─ (x: OnGrid, y: OnGrid, params=None)
 ─ (x: Continuous, y: object, params=None)
inverse:
 ─ (x: OnGrid, params=None)
```

## Details of implementations




---

## `__add__`

The `__add__` magic method is used to implement the `+` operator.

### `__add__(x: Linear, y: Linear, params=None)`

Sums two `Linear` fields, returning a new `Linear` field where the `params` are
the sum of the parameters of the two fields.

**Arguments**

| Name | Type | Description |
| :--- | :--- | :--- |
| `x` | `Linear` | The first field to be operated on |
| `y` | `Linear` | The second field to be operated on |
| `params` | `Optional[Dict[str, Any]]` | The parameters of the operator (unused) |

Default parameters: `None`

**Returns**

| Name | Type | Description |
| :--- | :--- | :--- |
| `y` | `Linear` | The result of the operation |

### `__add__(x: OnGrid, y: object, params=Params)`

### ` __add__(x: Continuous, y: Continuous, params=Params)`

### `__add__(x: Continuous, y: object, params=Params)`