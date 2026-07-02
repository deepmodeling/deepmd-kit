# Assembly Samples for Multi-Component Properties

`Assembly` is the user-facing name for samples whose label belongs to a set of related structures rather than to one frame.  Internally the training tensors still use `group_id.npy`, because the model aggregates frame embeddings by group, but user code should describe assemblies, samples, and components.

Core shape:

```python
from dpa_adapt import Assembly

a = Assembly(target="property_name", type_map=[...])
s = a.sample(key="sample_id", label=y, conditions={...})
s.add(coords, symbols, weight=w, pool_mask=..., role="...")
a.write("deepmd_data")
```

Each `sample` becomes one labeled training example.  Each component is one DeepMD frame.  The model computes frame embeddings, applies `pool_mask`, sums `weight * frame_embedding` within the sample, concatenates per-sample `conditions` as `fparam`, and predicts the target.

## Electrocatalysis Reaction Assemblies

Use this when the measured or computed label depends on several adsorbate states on the same catalyst surface.

- `Assembly.target`: `overpotential`, `activity`, `selectivity`, or a reaction free-energy descriptor.
- `sample`: one catalyst composition, slab, active site, or equation directory.
- `component`: clean slab and/or adsorbate states such as `*`, `H*`, `O*`, `OH*`, `OOH*`, `CO*`, `CHO*`.
- `weight`: stoichiometric coefficient or reaction-specific linear-combination coefficient; use `1.0` for pure learned state aggregation.
- `pool_mask`: exclude virtual adsorbate slots, cap atoms, or atoms intentionally not participating in the descriptor pool.
- `conditions`: pH, potential, electrolyte identity encoded upstream, surface coverage, temperature.

Example roles: `clean`, `O*`, `OH*`, `OOH*`.  The OER retrofit path is `Assembly.mark_existing(path, target="overpotential")`.

## Solvent and Electrolyte Mixtures

Use this when a formulation property belongs to a recipe rather than a single molecule.

- `Assembly.target`: `conductivity`, `viscosity`, `flash_point`, `solubility`, `dielectric_constant`.
- `sample`: one mixture recipe.
- `component`: solvent molecules, salt ion pairs, additives, cosolvents.
- `weight`: mole fraction, mass fraction, or normalized concentration contribution.
- `conditions`: temperature, total salt concentration, pH, pressure, measurement protocol flags.
- `pool_mask`: usually all atoms; exclude generated caps if fragments are represented by capped structures.

Suggested component roles: `solvent`, `salt`, `additive`, `cosolvent`.

## Polymer and Copolymer Properties

Use this when a polymer is represented by repeat units and end groups.

- `Assembly.target`: `cloud_point`, `glass_transition_temperature`, `lcst`, `solubility`, `modulus`.
- `sample`: one polymer or copolymer record.
- `component`: repeat units, start group, end group, optional side-chain fragments.
- `weight`: repeat-unit mole fraction; end-group share computed from `Mn` or supplied explicitly.
- `conditions`: standardized `Mn`, concentration, pH, salt concentrations, solvent identity.
- `pool_mask`: exclude artificial caps or attachment placeholders.

CSV ingestion remains available through `Assembly.from_polymer_csv(...).write(...)`; direct construction should use `Assembly.sample(...)` when the source is not the standard cloud-point CSV.

## Battery Cell Recipes

Use this when a performance label belongs to a full cell or formulation.

- `Assembly.target`: `cycle_life`, `capacity_retention`, `rate_capability`, `safety_window`, `formation_loss`.
- `sample`: one cell recipe and protocol.
- `component`: cathode active material, anode active material, electrolyte solvents, salts, additives, coating or binder fragments.
- `weight`: mass fraction, capacity-normalized contribution, or molar recipe fraction.
- `conditions`: C-rate, temperature, voltage window, state of charge, formation protocol.
- `pool_mask`: exclude caps/placeholders in fragment representations.

Suggested component roles: `cathode`, `anode`, `solvent`, `salt`, `additive`, `binder`.

## MOF/COF and Porous Framework Assemblies

Use this when a material property can be represented by structural building blocks plus optional guest molecules.

- `Assembly.target`: `uptake`, `selectivity`, `band_gap`, `stability`, `diffusivity`.
- `sample`: one framework topology or material entry.
- `component`: metal node, linker, functional group, counter-ion, guest molecule.
- `weight`: topology count, formula-normalized count, occupancy, or guest loading.
- `conditions`: temperature, pressure, guest identity encoding, humidity, activation protocol.
- `pool_mask`: exclude caps used to make linker/node fragments chemically valid.

Suggested component roles: `node`, `linker`, `functional_group`, `guest`.

## Naming Rationale

`Grouped` describes the tensor operation.  `Assembly` describes the scientific object users build: several components assembled into one labeled sample.  API names should therefore stay on the scientific layer (`Assembly`, `sample`, `component`, `weight`, `conditions`) while tensor files keep the implementation layer (`group_id`, `pool_mask`, `fparam`).
