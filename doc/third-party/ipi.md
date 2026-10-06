# Run path-integral MD with i-PI

> [!NOTE]
> See [Environment variables](../env.md) for the runtime environment variables.

The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` that computes the interactions (including energy, forces and virials). The server and client communicate via the Unix domain socket or the Internet socket. Installation instructions for i-PI can be found [here](../install/install-ipi.md). The client can be started by

```bash
i-pi input.xml &
dp_ipi water.json
```

It is noted that multiple instances of the client allow for computing, in parallel, the interactions of multiple replicas of the path-integral MD.

`water.json` is the parameter file for the client `dp_ipi`, and an example is provided:

```json
{
  "verbose": false,
  "use_unix": true,
  "port": 31415,
  "host": "localhost",
  "graph_file": "graph.pb",
  "coord_file": "conf.xyz",
  "atom_type": {
    "OW": 0,
    "HW1": 1,
    "HW2": 1
  }
}
```

The option **`use_unix`** is set to `true` to activate the Unix domain socket, otherwise, the Internet socket is used.

The option **`port`** should be the same as that in input.xml:

```xml
<port>31415</port>
```

The option **`graph_file`** provides the file name of the frozen model. The model can have either double or single float precision interface.

The `dp_ipi` gets the atom names from an [XYZ file](https://en.wikipedia.org/wiki/XYZ_file_format) provided by **`coord_file`** (meanwhile ignores all coordinates in it) and translates the names to atom types by rules provided by **`atom_type`**.

The client accepts the standard i-PI `INIT` message (replica ID, payload length,
and payload). Its model and atom types remain configured by the JSON input;
initialization payloads are consumed without a fixed size limit. Negative
payload lengths are rejected with an error.

The socket protocol uses atomic units: bohr for positions and cell vectors,
Hartree for energy and virial, and Hartree/bohr for forces. The driver converts
these to the model units internally. Forces are returned in the atom order
defined by the coordinate file, including when the model groups atoms by type.

The native `source/ipi/tests/test_driver_numeric.py` regression compares the
PyTorch inference API with `dp_ipi` over Unix and TCP sockets, including atom
permutations, periodic images and changing triclinic cells. Run it with a
PyTorch-enabled `dp_ipi` on `PATH` and the matching Python inference package.
