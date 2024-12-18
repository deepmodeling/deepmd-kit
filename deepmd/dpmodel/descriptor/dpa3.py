# SPDX-License-Identifier: LGPL-3.0-or-later


class RepFlowArgs:
    def __init__(
        self,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 20,
        axis_neuron: int = 4,
        update_angle: bool = True,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        skip_stat: bool = False,
    ) -> None:
        r"""The constructor for the RepFlowArgs class which defines the parameters of the repflow block in DPA3 descriptor.

        Parameters
        ----------
        n_dim : int, optional
            The dimension of node representation.
        e_dim : int, optional
            The dimension of edge representation.
        a_dim : int, optional
            The dimension of angle representation.
        nlayers : int, optional
            Number of repflow layers.
        e_rcut : float, optional
            The edge cut-off radius.
        e_rcut_smth : float, optional
            Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
        e_sel : int, optional
            Maximally possible number of selected edge neighbors.
        a_rcut : float, optional
            The angle cut-off radius.
        a_rcut_smth : float, optional
            Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
        a_sel : int, optional
            Maximally possible number of selected angle neighbors.
        axis_neuron : int, optional
            The number of dimension of submatrix in the symmetrization ops.
        update_angle : bool, optional
            Where to update the angle rep. If not, only node and edge rep will be used.
        update_style : str, optional
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
            and `update_residual_init`.
        update_residual : float, optional
            When update using residual mode, the initial std of residual vector weights.
        update_residual_init : str, optional
            When update using residual mode, the initialization mode of residual vector weights.
        """
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.skip_stat = skip_stat

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "nlayers": self.nlayers,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "axis_neuron": self.axis_neuron,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowArgs":
        return cls(**data)
