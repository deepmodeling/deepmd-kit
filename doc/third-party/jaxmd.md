# Run MD with JAX-MD

:::{note}
See [Environment variables](../env.md) for the runtime environment variables.
:::

DeePMD-kit provides a JAX-MD compatible interface for DeePMD models trained with
the JAX backend. The interface adapts a DeePMD model to the usual JAX-MD style,
where a neighbor-list factory and an energy function are passed to JAX-MD
simulation routines.

The interface is available from `deepmd.jax.jax_md`.

## Requirements

Install DeePMD-kit with the JAX backend and install
[JAX-MD](https://github.com/jax-md/jax-md). The JAX-MD package is an optional
runtime dependency and is not required for other DeePMD-kit interfaces.

## Basic usage

The most common entry point is `as_jax_md`, which returns a JAX-MD neighbor-list
function and a potential energy function:

```python
import jax
import jax.numpy as jnp
from jax_md import space

from deepmd.jax.jax_md import as_jax_md

box = jnp.asarray([12.4447, 12.4447, 12.4447])
coord = jnp.asarray(...)  # shape: (natoms, 3)
atom_types = jnp.asarray(...)  # shape: (natoms,), DeePMD type indexes

displacement_fn, shift_fn = space.periodic(box)
neighbor_fn, potential_fn = as_jax_md(
    "model.ckpt.jax",
    displacement_fn,
    box,
    atom_types,
    dr_threshold=0.2,
    capacity_multiplier=1.5,
)

neighbor = neighbor_fn.allocate(coord)
energy = potential_fn(coord, neighbor=neighbor)
force = -jax.grad(lambda x: potential_fn(x, neighbor=neighbor))(coord)
```

The returned `potential_fn` accepts a single-frame coordinate array with shape
`(natoms, 3)` and returns the scalar total energy. The optional `neighbor`
argument should be a dense JAX-MD neighbor list allocated by the returned
`neighbor_fn`.

## Running dynamics

The potential can be used with JAX-MD simulation routines. A minimal NVE loop
looks like:

```python
from jax_md import simulate

K_B_EV_PER_K = 8.617333262145e-5
kT = K_B_EV_PER_K * 330.0
mass = jnp.ones((coord.shape[0], 1))

init_fn, step_fn = simulate.nve(potential_fn, shift_fn, dt=0.0005)
state = init_fn(jax.random.key(0), coord, kT=kT, mass=mass, neighbor=neighbor)

for _ in range(10):
    neighbor = neighbor_fn.update(state.position, neighbor)
    state = step_fn(state, neighbor=neighbor)
```

For a complete water example using the same 192-atom configuration as the
LAMMPS example, see `examples/water/jax_md`.

## Model files

`deepmd.jax.jax_md.load_model` accepts:

- a DeePMD JAX checkpoint path ending in `.jax`,
- a DeePMD HLO model path ending in `.hlo`,
- an already constructed JAX DeePMD model object.

The `atom_types` argument may be an integer array of DeePMD type indexes. It
may also be a sequence of type names if the model has a `type_map`.

## Neighbor lists

The helper `neighbor_list` creates a dense JAX-MD neighbor-list function using
the model cutoff:

```python
from deepmd.jax.jax_md import energy_fn, neighbor_list

neighbor_fn = neighbor_list("model.ckpt.jax", displacement_fn, box)
potential_fn = energy_fn(
    "model.ckpt.jax",
    atom_types,
    box=box,
    displacement_fn=displacement_fn,
)
```

Only dense JAX-MD neighbor lists are currently supported. If the neighbor-list
buffer overflows during a simulation, increase `capacity_multiplier` or rebuild
the neighbor list with a larger capacity.

## Units

The JAX-MD interface does not perform unit conversion. Coordinates, box
vectors, energies, forces, masses, and timesteps should be provided in units
consistent with the DeePMD model and the chosen JAX-MD simulation setup.
