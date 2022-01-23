import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from typing import Callable

def _unroll(x):
  return tuple([x.replace_params(y) for y in x.params])

def euler_integration(f, x0, dt, output_steps):
  #assert any(map(lambda x: x >= 0, output_steps))
  
  def euler_step(i, x):
    dx_dt = f(x, i * dt)
    return x + dt*dx_dt

  def euler_jump(x_t, i):
    x = x_t[0]
    start = x_t[1]
    end = start + i

    y = jax.lax.fori_loop(start, end, euler_step, x)
    return (y, end), y

  jumps = jnp.diff(output_steps)

  _, ys = jax.lax.scan(euler_jump, (x0, 0.0), jumps)
  return _unroll(ys)