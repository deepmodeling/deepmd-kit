# Install i-PI
The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` that computes the interactions (including energy, force and virial). The server and client communicates via the Unix domain socket or the Internet socket. A full instruction of i-PI can be found [here](http://ipi-code.org/). The source code and a complete installation instructions of i-PI can be found [here](https://github.com/i-pi/i-pi).
To use i-PI with already existing drivers, install and update using Pip:
```bash
pip install -U i-PI
```

Test with Pytest:
```bash
pip install pytest
pytest --pyargs ipi.tests
```