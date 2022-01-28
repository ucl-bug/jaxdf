# `jaxdf.operators.dummy`

---

## `dummy`

A dummy operator, useful for testing and debugging

### `dummy(x: OnGrid, params=None)`

Generates a new field with parameters 

```python
new_params = x.params + params['k']
```

**Arguments**

| Name | Type | Description |
| :--- | :--- | :--- |
| `x` | `OnGrid` | The field to be operated on |
| `params` | `Optional[Dict[str, Any]]` | The parameters of the operator |

Default parameters:
```python
params = {"k": 3}
```

**Returns**

| Name | Type | Description |
| :--- | :--- | :--- |
| `y` | `OnGrid` | The result of the operation |

### `dummy(x: Continuous, params=Params)`

Generates a new field with the same parameters, and `get_fun` given by
```python
get_x = x.aux['get_field']
get_fun = lambda p, x: get_x(p, x) + params['k']
```

**Arguments**
| Name | Type | Description |
| :--- | :--- | :--- |
| `x` | `Continuous` | The field to be operated on |
| `params` | `Params` | The parameters of the operator |

Default parameters:
```python
params = {"k": 3}
```

**Returns**
| Name | Type | Description |
| :--- | :--- | :--- |
| `y` | `Continuous` | The result of the operation |