### Run path-integral MD with i-PI
The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` (or `dp_ipi_low` for low precision) that computes the interactions (including energy, force and virial). The server and client communicates via the Unix domain socket or the Internet socket. Installation instructions of i-PI can be found [here](../install/install-ipi.md). The client can be started by
```bash
i-pi input.xml &
dp_ipi water.json
```
It is noted that multiple instances of the client is allow for computing, in parallel, the interactions of multiple replica of the path-integral MD.

`water.json` is the parameter file for the client `dp_ipi`, and an example is provided:
```json
{
    "verbose":		false,
    "use_unix":		true,
    "port":		31415,
    "host":		"localhost",
    "graph_file":	"graph.pb",
    "coord_file":	"conf.xyz",
    "atom_type" : {
	"OW":		0, 
	"HW1":		1,
	"HW2":		1
    }
}
```
The option **`use_unix`** is set to `true` to activate the Unix domain socket, otherwise, the Internet socket is used.

The option **`port`** should be the same as that in input.xml:
```xml
<port>31415</port>
```

The option **`graph_file`** provides the file name of the frozen model.

The `dp_ipi` gets the atom names from an [XYZ file](https://en.wikipedia.org/wiki/XYZ_file_format) provided by **`coord_file`** (meanwhile ignores all coordinates in it), and translates the names to atom types by rules provided by **`atom_type`**.