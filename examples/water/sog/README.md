# Input for the SOG model

This directory provides a SOG training example based on the same water data split and training layout used by `examples/water/dpa3`.

## Run

```bash
cd examples/water/sog
dp --pt train input_torch.json --skip-neighbor-stat
```

## Notes

- Descriptor: DPA3 (`model.descriptor.type = "dpa3"`)
- Fitting: SOG energy (`model.fitting_net.type = "sog_energy"`)
- Data systems are reused from `examples/water/data/data_0` to `data_3`.
