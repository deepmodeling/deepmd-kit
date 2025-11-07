# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
    Optional,
    Union,
)

import numpy as np
import paddle

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pd.model.network.mlp import (
    FittingNet,
    NetworkCollection,
)
from deepmd.pd.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pd.utils.exclude_mask import (
    AtomExcludeMask,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
    to_paddle_tensor,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class Fitting(paddle.nn.Layer, BaseFitting):
    # plugin moved to BaseFitting

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            return BaseFitting.__new__(BaseFitting, *args, **kwargs)
        return super().__new__(cls)

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only fitting nets of the same type can share params!"
        )
        if shared_level == 0:
            # only not share the bias_atom_e and the case_embd
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._sub_layers:
                self._sub_layers[item] = base_class._sub_layers[item]
        else:
            raise NotImplementedError

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        protection: float = 1e-2,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the fittings from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        protection : float
            Divided-by-zero protection
        """
        if self.numb_fparam == 0 and self.numb_aparam == 0:
            # skip data statistics
            return
        if callable(merged):
            sampled = merged()
        else:
            sampled = merged
        # stat fparam
        if self.numb_fparam > 0:
            cat_data = paddle.concat([frame["fparam"] for frame in sampled], axis=0)
            cat_data = paddle.reshape(cat_data, [-1, self.numb_fparam])
            fparam_avg = paddle.mean(cat_data, axis=0)
            fparam_std = paddle.std(cat_data, axis=0, unbiased=False)
            fparam_std = paddle.where(
                fparam_std < protection,
                paddle.to_tensor(protection, dtype=fparam_std.dtype),
                fparam_std,
            )
            fparam_inv_std = 1.0 / fparam_std
            paddle.assign(
                paddle.to_tensor(
                    fparam_avg, place=env.DEVICE, dtype=self.fparam_avg.dtype
                ),
                self.fparam_avg,
            )
            paddle.assign(
                paddle.to_tensor(
                    fparam_inv_std, place=env.DEVICE, dtype=self.fparam_inv_std.dtype
                ),
                self.fparam_inv_std,
            )
        # stat aparam
        if self.numb_aparam > 0:
            sys_sumv = []
            sys_sumv2 = []
            sys_sumn = []
            for ss_ in [frame["aparam"] for frame in sampled]:
                ss = paddle.reshape(ss_, [-1, self.numb_aparam])
                sys_sumv.append(paddle.sum(ss, axis=0))
                sys_sumv2.append(paddle.sum(ss * ss, axis=0))
                sys_sumn.append(ss.shape[0])
            sumv = paddle.sum(paddle.stack(sys_sumv), axis=0)
            sumv2 = paddle.sum(paddle.stack(sys_sumv2), axis=0)
            sumn = sum(sys_sumn)
            aparam_avg = sumv / sumn
            aparam_std = paddle.sqrt(sumv2 / sumn - (sumv / sumn) ** 2)
            aparam_std = paddle.where(
                aparam_std < protection,
                paddle.to_tensor(
                    protection, dtype=aparam_std.dtype, place=aparam_std.device
                ),
                aparam_std,
            )
            aparam_inv_std = 1.0 / aparam_std
            paddle.assign(
                paddle.to_tensor(
                    aparam_avg, place=env.DEVICE, dtype=self.aparam_avg.dtype
                ),
                self.aparam_avg,
            )
            paddle.assign(
                paddle.to_tensor(
                    aparam_inv_std, place=env.DEVICE, dtype=self.aparam_inv_std.dtype
                ),
                self.aparam_inv_std,
            )


class GeneralFitting(Fitting):
    """Construct a general fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'energy', 'dipole', and 'polar'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int
        The output dimension of the fitting net.
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_e : paddle.Tensor, optional
        Average energy per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    default_fparam: list[float], optional
        The default frame parameter. If set, when `fparam.npy` files are not included in the data system,
        this value will be used as the default value for the frame parameter in the fitting net.
        This parameter is not supported in PaddlePaddle.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    exclude_types: list[int]
        Atomic contributions of the excluded atom types are set zero.
    trainable : Union[list[bool], bool]
        If the parameters in the fitting net are trainable.
        Now this only supports setting all the parameters in the fitting net at one state.
        When in list[bool], the trainable will be True only if all the boolean parameters are True.
    remove_vaccum_contribution: list[bool], optional
        Remove vacuum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vacuum contribution for the atom types in the list.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    use_aparam_as_mask: bool
        If True, the aparam will not be used in fitting net for embedding.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: Optional[paddle.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        trainable: Union[bool, list[bool]] = True,
        remove_vaccum_contribution: Optional[list[bool]] = None,
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        default_fparam: Optional[list[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.neuron = neuron
        self.mixed_types = mixed_types
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.register_buffer(
            "buffer_numb_fparam", paddle.to_tensor([numb_fparam], dtype=paddle.int64)
        )
        self.numb_aparam = numb_aparam
        self.register_buffer(
            "buffer_numb_aparam", paddle.to_tensor([numb_aparam], dtype=paddle.int64)
        )
        self.dim_case_embd = dim_case_embd
        self.default_fparam = default_fparam
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.rcond = rcond
        self.seed = seed
        self.type_map = type_map
        if type_map is not None:
            self.register_buffer(
                "buffer_type_map",
                paddle.to_tensor([ord(c) for c in " ".join(self.type_map)]),
            )
        self.use_aparam_as_mask = use_aparam_as_mask
        # order matters, should be place after the assignment of ntypes
        self.reinit_exclude(exclude_types)
        self.trainable = trainable
        # need support for each layer settings
        self.trainable = (
            all(self.trainable) if isinstance(self.trainable, list) else self.trainable
        )
        self.remove_vaccum_contribution = remove_vaccum_contribution

        net_dim_out = self._net_out_dim()
        # init constants
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, net_dim_out], dtype=np.float64)
        bias_atom_e = paddle.to_tensor(
            bias_atom_e, dtype=env.GLOBAL_PD_FLOAT_PRECISION, place=device
        )
        bias_atom_e = bias_atom_e.reshape([self.ntypes, net_dim_out])
        if not self.mixed_types:
            assert self.ntypes == bias_atom_e.shape[0], "Element count mismatches!"
        self.register_buffer("bias_atom_e", bias_atom_e)

        if self.numb_fparam > 0:
            self.register_buffer(
                "fparam_avg",
                paddle.zeros([self.numb_fparam], dtype=self.prec).to(device=device),
            )
            self.register_buffer(
                "fparam_inv_std",
                paddle.ones([self.numb_fparam], dtype=self.prec).to(device=device),
            )
        else:
            self.fparam_avg, self.fparam_inv_std = None, None
        if self.numb_aparam > 0:
            self.register_buffer(
                "aparam_avg",
                paddle.zeros([self.numb_aparam], dtype=self.prec).to(device=device),
            )
            self.register_buffer(
                "aparam_inv_std",
                paddle.ones([self.numb_aparam], dtype=self.prec).to(device=device),
            )
        else:
            self.aparam_avg, self.aparam_inv_std = None, None

        if self.dim_case_embd > 0:
            self.register_buffer(
                "case_embd",
                paddle.zeros(self.dim_case_embd, dtype=self.prec).to(device=device),
                # paddle.eye(self.dim_case_embd, dtype=self.prec).to(device=device)[0],
            )
        else:
            self.case_embd = None

        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )

        self.filter_layers = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    net_dim_out,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )
        # set trainable
        for param in self.parameters():
            param.stop_gradient = not self.trainable

        self.eval_return_middle_output = False

    def reinit_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        assert self.mixed_types, "Only models in mixed types can perform type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.reinit_exclude(map_atom_exclude_types(self.exclude_types, remap_index))
        if has_new_type:
            extend_shape = [len(type_map), *list(self.bias_atom_e.shape[1:])]
            extend_bias_atom_e = paddle.zeros(
                extend_shape,
                dtype=self.bias_atom_e.dtype,
            ).to(device=self.bias_atom_e.place)
            self.bias_atom_e = paddle.concat(
                [self.bias_atom_e, extend_bias_atom_e], axis=0
            )
        self.bias_atom_e = self.bias_atom_e[remap_index]

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "Fitting",
            "@version": 4,
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "default_fparam": self.default_fparam,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "nets": self.filter_layers.serialize(),
            "rcond": self.rcond,
            "exclude_types": self.exclude_types,
            "@variables": {
                "bias_atom_e": to_numpy_array(self.bias_atom_e),
                "case_embd": to_numpy_array(self.case_embd),
                "fparam_avg": to_numpy_array(self.fparam_avg),
                "fparam_inv_std": to_numpy_array(self.fparam_inv_std),
                "aparam_avg": to_numpy_array(self.aparam_avg),
                "aparam_inv_std": to_numpy_array(self.aparam_inv_std),
            },
            "type_map": self.type_map,
            # "tot_ener_zero": self.tot_ener_zero ,
            # "trainable": self.trainable ,
            # "atom_ener": self.atom_ener ,
            # "layer_name": self.layer_name ,
            # "spin": self.spin ,
            ## NOTICE:  not supported by far
            "tot_ener_zero": False,
            "trainable": [self.trainable] * (len(self.neuron) + 1),
            "layer_name": None,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        variables = data.pop("@variables")
        nets = data.pop("nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_paddle_tensor(variables[kk])
        obj.filter_layers = NetworkCollection.deserialize(nets)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.numb_aparam

    def get_buffer_dim_fparam(self) -> paddle.Tensor:
        """Get the number (dimension) of frame parameters of this atomic model as a buffer-style Tensor."""
        return self.buffer_numb_fparam

    def get_buffer_dim_aparam(self) -> paddle.Tensor:
        """Get the number (dimension) of atomic parameters of this atomic model as a buffer-style Tensor."""
        return self.buffer_numb_aparam

    # make jit happy
    exclude_types: list[int]

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        # make jit happy
        sel_type: list[int] = []
        for ii in range(self.ntypes):
            if ii not in self.exclude_types:
                sel_type.append(ii)
        return sel_type

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_buffer_type_map(self) -> paddle.Tensor:
        """
        Return the type map as a buffer-style Tensor for JIT saving.

        The original type map (e.g., ['Ni', 'O']) is first joined into a single space-separated string
        (e.g., "Ni O"). Each character in this string is then converted to its ASCII code using `ord()`,
        and the resulting integer sequence is stored as a 1D paddle.Tensor of dtype int.

        This format allows the type map to be serialized as a raw byte buffer during JIT model saving.
        """
        return self.buffer_type_map

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this fitting net by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.case_embd = paddle.eye(self.dim_case_embd, dtype=self.prec).to(device)[
            case_idx
        ]

    def set_return_middle_output(self, return_middle_output: bool = True) -> None:
        self.eval_return_middle_output = return_middle_output

    def __setitem__(self, key, value) -> None:
        if key in ["bias_atom_e"]:
            value = value.reshape([self.ntypes, self._net_out_dim()])
            self.bias_atom_e = value
        elif key in ["fparam_avg"]:
            self.fparam_avg = value
        elif key in ["fparam_inv_std"]:
            self.fparam_inv_std = value
        elif key in ["aparam_avg"]:
            self.aparam_avg = value
        elif key in ["aparam_inv_std"]:
            self.aparam_inv_std = value
        elif key in ["case_embd"]:
            self.case_embd = value
        elif key in ["scale"]:
            self.scale = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ["bias_atom_e"]:
            return self.bias_atom_e
        elif key in ["fparam_avg"]:
            return self.fparam_avg
        elif key in ["fparam_inv_std"]:
            return self.fparam_inv_std
        elif key in ["aparam_avg"]:
            return self.aparam_avg
        elif key in ["aparam_inv_std"]:
            return self.aparam_inv_std
        elif key in ["case_embd"]:
            return self.case_embd
        elif key in ["scale"]:
            return self.scale
        else:
            raise KeyError(key)

    @abstractmethod
    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        pass

    def _extend_f_avg_std(self, xx: paddle.Tensor, nb: int) -> paddle.Tensor:
        return paddle.tile(xx.reshape([1, self.numb_fparam]), [nb, 1])

    def _extend_a_avg_std(self, xx: paddle.Tensor, nb: int, nloc: int) -> paddle.Tensor:
        return paddle.tile(xx.reshape([1, 1, self.numb_aparam]), [nb, nloc, 1])

    def _forward_common(
        self,
        descriptor: paddle.Tensor,
        atype: paddle.Tensor,
        gr: Optional[paddle.Tensor] = None,
        g2: Optional[paddle.Tensor] = None,
        h2: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
    ):
        # cast the input to internal precsion
        xx = descriptor.astype(self.prec)
        fparam = fparam.astype(self.prec) if fparam is not None else None
        aparam = aparam.astype(self.prec) if aparam is not None else None

        if self.remove_vaccum_contribution is not None:
            # TODO: compute the input for vaccm when remove_vaccum_contribution is set
            # Ideally, the input for vacuum should be computed;
            # we consider it as always zero for convenience.
            # Needs a compute_input_stats for vacuum passed from the
            # descriptor.
            xx_zeros = paddle.zeros_like(xx)
        else:
            xx_zeros = None
        nf, nloc, nd = xx.shape
        net_dim_out = self._net_out_dim()

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
            )
        # check fparam dim, concate to input descriptor
        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    "get an input fparam of dim {fparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_fparam}.",
                )
            fparam = fparam.reshape([nf, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = paddle.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = paddle.concat(
                [xx, fparam],
                axis=-1,
            )
            if xx_zeros is not None:
                xx_zeros = paddle.concat(
                    [xx_zeros, fparam],
                    axis=-1,
                )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    f"get an input aparam of dim {aparam.shape[-1]}, ",
                    f"which is not consistent with {self.numb_aparam}.",
                )
            aparam = aparam.reshape([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = paddle.concat(
                [xx, aparam],
                axis=-1,
            )
            if xx_zeros is not None:
                xx_zeros = paddle.concat(
                    [xx_zeros, aparam],
                    axis=-1,
                )

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = paddle.tile(self.case_embd.reshape([1, 1, -1]), [nf, nloc, 1])
            xx = paddle.concat(
                [xx, case_embd],
                axis=-1,
            )
            if xx_zeros is not None:
                xx_zeros = paddle.concat(
                    [xx_zeros, case_embd],
                    axis=-1,
                )

        outs = paddle.zeros(
            (nf, nloc, net_dim_out),
            dtype=env.GLOBAL_PD_FLOAT_PRECISION,
        )
        results = {}

        if self.mixed_types:
            atom_property = self.filter_layers.networks[0](xx)
            if self.eval_return_middle_output:
                results["middle_output"] = self.filter_layers.networks[
                    0
                ].call_until_last(xx)
            if xx_zeros is not None:
                atom_property -= self.filter_layers.networks[0](xx_zeros)
            outs = (
                outs + atom_property + self.bias_atom_e[atype].astype(self.prec)
            )  # Shape is [nframes, natoms[0], net_dim_out]
        else:
            if self.eval_return_middle_output:
                outs_middle = paddle.zeros(
                    (nf, nloc, self.neuron[-1]),
                    dtype=self.prec,
                ).to(device=descriptor.place)  # jit assertion
                for type_i, ll in enumerate(self.filter_layers.networks):
                    mask = (atype == type_i).unsqueeze(-1)
                    mask = paddle.tile(mask, (1, 1, net_dim_out))
                    middle_output_type = ll.call_until_last(xx)
                    middle_output_type = paddle.where(
                        paddle.tile(mask, (1, 1, self.neuron[-1])),
                        middle_output_type,
                        paddle.zeros_like(middle_output_type),
                    )
                    outs_middle = outs_middle + middle_output_type
                results["middle_output"] = outs_middle
            for type_i, ll in enumerate(self.filter_layers.networks):
                mask = (atype == type_i).unsqueeze(-1)
                mask.stop_gradient = True
                mask = paddle.tile(mask, (1, 1, net_dim_out))
                atom_property = ll(xx)
                if xx_zeros is not None:
                    # must assert, otherwise jit is not happy
                    assert self.remove_vaccum_contribution is not None
                    if not (
                        len(self.remove_vaccum_contribution) > type_i
                        and not self.remove_vaccum_contribution[type_i]
                    ):
                        atom_property -= ll(xx_zeros)
                atom_property = atom_property + self.bias_atom_e[type_i]
                atom_property = paddle.where(
                    mask, atom_property, paddle.full_like(atom_property, 0.0)
                )
                outs = (
                    outs + atom_property
                )  # Shape is [nframes, natoms[0], net_dim_out]
        # nf x nloc
        mask = self.emask(atype).astype("bool")
        # nf x nloc x nod
        outs = paddle.where(mask[:, :, None], outs, paddle.zeros_like(outs))
        results.update({self.var_name: outs})
        return results
