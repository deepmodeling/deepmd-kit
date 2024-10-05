# Install i-PI

The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` that computes the interactions (including energy, forces and virials). The server and client communicate via the Unix domain socket or the Internet socket. Full documentation for i-PI can be found [here](http://ipi-code.org/). The source code and a complete installation guide for i-PI can be found [here](https://github.com/i-pi/i-pi).
To use i-PI with already existing drivers, install and update using Pip:

```bash
pip install -U ipi
```

Test with Pytest:

```bash
pip install pytest
pytest --pyargs ipi.tests
```
