# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Uni-Mol model on the PyTorch-Exportable backend.

Checks that the registered path builds, that it agrees with the array-API
implementation on the same weights, and that the objective still reproduces the
values dumped from upstream. The golden archive and how it was produced are
described in ``source/tests/common/dpmodel/test_unimol.py``.
"""

import os
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.unimol import (
    UNIMOL_ELEMENTS,
    DescrptUniMol as DescrptUniMolDP,
    unimol_vocabulary,
)
from deepmd.dpmodel.fitting.unimol_pretrain import (
    UniMolPretrainFitting as UniMolPretrainFittingDP,
)
from deepmd.pt_expt.loss.unimol import (
    UniMolLoss,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.utils.argcheck import (
    normalize,
)

GOLDEN = os.path.join(
    os.path.dirname(__file__), "..", "..", "common", "dpmodel", "unimol_v1_golden.npz"
)
SMALL = {"layers": 2, "dim": 32, "ffn": 64, "heads": 4, "vocab": 31}


def _assign(module, name, array):
    """Put a NumPy array into a torch-backed slot, keeping its kind."""
    current = getattr(module, name)
    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=current.dtype)
    if isinstance(current, torch.nn.Parameter):
        tensor = torch.nn.Parameter(tensor, requires_grad=current.requires_grad)
    setattr(module, name, tensor)


class TestUniMolPtExpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = np.load(GOLDEN)
        cls.weights = {
            k[len("small_weights/") :]: v
            for k, v in cls.golden.items()
            if k.startswith("small_weights/")
        }
        cls.n_real = (cls.golden["input/src_tokens"] != 0).sum(axis=1) - 2
        cls.type_map = [*UNIMOL_ELEMENTS, "[MASK]"]

    def inputs(self):
        vocab = unimol_vocabulary()
        type_of = {sym: i for i, sym in enumerate(self.type_map)}
        src_tokens = self.golden["input/src_tokens"]
        src_coord = self.golden["input/src_coord"].astype(np.float64)
        nf, nloc = len(self.n_real), int(self.n_real.max())
        coord = np.zeros((nf, nloc, 3))
        atype = np.zeros((nf, nloc), dtype=np.int64)
        nlist = np.full((nf, nloc, nloc - 1), -1, dtype=np.int64)
        for f in range(nf):
            n = int(self.n_real[f])
            coord[f, :n] = src_coord[f, 1 : n + 1]
            atype[f, :n] = [type_of[vocab[t]] for t in src_tokens[f, 1 : n + 1]]
            for i in range(n):
                others = [j for j in range(n) if j != i]
                nlist[f, i, : len(others)] = others
        return coord, atype, nlist

    def config(self, nloc, single_precision_basis: bool = True):
        return {
            "model": {
                "type_map": self.type_map,
                "descriptor": {
                    "type": "unimol",
                    "encoder_layers": SMALL["layers"],
                    "encoder_embed_dim": SMALL["dim"],
                    "encoder_ffn_embed_dim": SMALL["ffn"],
                    "encoder_attention_heads": SMALL["heads"],
                    "max_atoms": nloc,
                    "virtual_token_position": "origin",
                    "single_precision_basis": single_precision_basis,
                    "precision": "float64",
                },
                "fitting_net": {
                    "type": "unimol_pretrain",
                    "n_token": SMALL["vocab"],
                    "attention_heads": SMALL["heads"],
                    "max_atoms": nloc,
                    "precision": "float64",
                },
            },
            "learning_rate": {"type": "exp", "start_lr": 1e-4, "stop_lr": 1e-6},
            "loss": {"type": "unimol"},
            "training": {
                "training_data": {"systems": ["x"]},
                "numb_steps": 1,
                "seed": 1,
            },
        }

    def load_weights(self, descriptor, fitting, torch_backend: bool) -> None:
        w = self.weights
        put = (
            _assign
            if torch_backend
            else (lambda m, n, a: setattr(m, n, np.ascontiguousarray(a)))
        )

        def lin(layer, wkey, bkey):
            put(layer, "w", w[wkey].T)
            if bkey in w and layer.b is not None:
                put(layer, "b", w[bkey])

        def ln(layer, prefix):
            put(layer, "w", w[prefix + ".weight"])
            put(layer, "b", w[prefix + ".bias"])

        put(descriptor, "embed_tokens", w["embed_tokens.weight"])
        for key in ("means", "stds", "mul", "bias"):
            put(descriptor.gbf, key, w[f"gbf.{key}.weight"])
        lin(
            descriptor.gbf_proj.linear1,
            "gbf_proj.linear1.weight",
            "gbf_proj.linear1.bias",
        )
        lin(
            descriptor.gbf_proj.linear2,
            "gbf_proj.linear2.weight",
            "gbf_proj.linear2.bias",
        )
        ln(descriptor.encoder.emb_layer_norm, "encoder.emb_layer_norm")
        ln(descriptor.encoder.final_layer_norm, "encoder.final_layer_norm")
        ln(descriptor.encoder.final_head_layer_norm, "encoder.final_head_layer_norm")
        for i, layer in enumerate(descriptor.encoder.layers):
            p = f"encoder.layers.{i}."
            lin(
                layer.self_attn.in_proj,
                p + "self_attn.in_proj.weight",
                p + "self_attn.in_proj.bias",
            )
            lin(
                layer.self_attn.out_proj,
                p + "self_attn.out_proj.weight",
                p + "self_attn.out_proj.bias",
            )
            ln(layer.self_attn_layer_norm, p + "self_attn_layer_norm")
            lin(layer.fc1, p + "fc1.weight", p + "fc1.bias")
            lin(layer.fc2, p + "fc2.weight", p + "fc2.bias")
            ln(layer.final_layer_norm, p + "final_layer_norm")
        lin(fitting.lm_head.dense, "lm_head.dense.weight", "lm_head.dense.bias")
        ln(fitting.lm_head.layer_norm, "lm_head.layer_norm")
        lin(fitting.lm_head.out_proj, "lm_head.weight", "lm_head.bias")
        lin(
            fitting.pair2coord_proj.linear1,
            "pair2coord_proj.linear1.weight",
            "pair2coord_proj.linear1.bias",
        )
        lin(
            fitting.pair2coord_proj.linear2,
            "pair2coord_proj.linear2.weight",
            "pair2coord_proj.linear2.bias",
        )
        lin(fitting.dist_head.dense, "dist_head.dense.weight", "dist_head.dense.bias")
        ln(fitting.dist_head.layer_norm, "dist_head.layer_norm")
        lin(
            fitting.dist_head.out_proj,
            "dist_head.out_proj.weight",
            "dist_head.out_proj.bias",
        )

    def build_torch_model(self, nloc, single_precision_basis: bool = True):
        model = get_model(normalize(self.config(nloc, single_precision_basis))["model"])
        self.load_weights(
            model.atomic_model.descriptor,
            model.atomic_model.fitting_net,
            torch_backend=True,
        )
        model.eval()
        return model

    def test_registered_path_builds(self) -> None:
        coord, _, _ = self.inputs()
        model = self.build_torch_model(coord.shape[1])
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(type(model).__name__, "UniMolPretrainModel")
        self.assertEqual(
            sorted(model.translated_output_def().keys()),
            [
                "coord_update",
                "delta_pair_norm",
                "mask",
                "pair_dist",
                "token_logits",
                "x_norm",
            ],
        )

    def test_matches_the_array_api_implementation(self) -> None:
        """Same weights, same numbers, once the fp32 basis is out of the way.

        Upstream evaluates the Gaussian basis in fp32, and NumPy and Torch round
        that last place differently, which is worth about 2e-8 relative on the
        node representation. Turning that off isolates the two implementations
        from each other, and then they agree to fp64 rounding.
        """
        coord, atype, nlist = self.inputs()
        nloc = coord.shape[1]
        model = self.build_torch_model(nloc, single_precision_basis=False)
        ret = model.forward_lower(
            torch.as_tensor(coord.reshape(len(self.n_real), -1)),
            torch.as_tensor(atype),
            torch.as_tensor(nlist),
            None,
        )

        cfg = normalize(self.config(nloc, single_precision_basis=False))["model"]
        descriptor = DescrptUniMolDP(
            type_map=self.type_map,
            **{k: v for k, v in cfg["descriptor"].items() if k != "type"},
        )
        fitting = UniMolPretrainFittingDP(
            ntypes=len(self.type_map),
            dim_descrpt=SMALL["dim"],
            **{k: v for k, v in cfg["fitting_net"].items() if k != "type"},
        )
        self.load_weights(descriptor, fitting, torch_backend=False)
        reference = fitting.call_tokens(descriptor.forward_tokens(coord, atype, nlist))

        for key in (
            "token_logits",
            "coord_update",
            "pair_dist",
            "x_norm",
            "delta_pair_norm",
        ):
            with self.subTest(output=key):
                np.testing.assert_allclose(
                    ret[key].detach().cpu().numpy(),
                    np.asarray(reference[key]),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_single_precision_basis_costs_one_fp32_place(self) -> None:
        """With upstream's fp32 basis the two backends part company measurably.

        The gap is one fp32 unit in the last place in the basis, amplified by
        the stack. It is recorded here so that the looser tolerance elsewhere
        has a stated cause rather than being tuned until tests pass.
        """
        coord, atype, nlist = self.inputs()
        nloc = coord.shape[1]
        exact = self.build_torch_model(nloc, single_precision_basis=False)
        upstream_like = self.build_torch_model(nloc, single_precision_basis=True)
        args = (
            torch.as_tensor(coord.reshape(len(self.n_real), -1)),
            torch.as_tensor(atype),
            torch.as_tensor(nlist),
            None,
        )
        a = exact.forward_lower(*args)["token_logits"].detach().cpu().numpy()
        b = upstream_like.forward_lower(*args)["token_logits"].detach().cpu().numpy()
        gap = np.abs(a - b).max() / np.abs(a).max()
        self.assertLess(gap, 1e-5)
        self.assertGreater(gap, 1e-12)

    def test_objective_matches_upstream(self) -> None:
        coord, atype, nlist = self.inputs()
        nloc = coord.shape[1]
        model = self.build_torch_model(nloc)
        ret = model.forward_lower(
            torch.as_tensor(coord.reshape(len(self.n_real), -1)),
            torch.as_tensor(atype),
            torch.as_tensor(nlist),
            None,
        )
        ncol = model.atomic_model.fitting_net.max_atoms + 2
        nf = len(self.n_real)
        labels = {
            "unimol_token_target": torch.zeros((nf, nloc), dtype=torch.int64),
            "unimol_coord_target": torch.zeros((nf, nloc, 3), dtype=torch.float64),
            "unimol_dist_target": torch.zeros((nf, nloc, ncol), dtype=torch.float64),
            "unimol_token_mask": torch.zeros((nf, ncol), dtype=torch.int64),
        }
        for f in range(nf):
            n = int(self.n_real[f])
            labels["unimol_token_target"][f, :n] = torch.as_tensor(
                self.golden["target/tokens_target"][f, 1 : n + 1].astype(np.int64)
            )
            labels["unimol_coord_target"][f, :n] = torch.as_tensor(
                self.golden["target/coord_target"][f, 1 : n + 1].astype(np.float64)
            )
            labels["unimol_dist_target"][f, :n, : n + 2] = torch.as_tensor(
                self.golden["target/distance_target"][f, 1 : n + 1, : n + 2].astype(
                    np.float64
                )
            )
            labels["unimol_token_mask"][f, : n + 2] = 1
        labels = {k: v.to(ret["token_logits"].device) for k, v in labels.items()}

        total, more = UniMolLoss().call(1.0, 0, ret, labels)
        expected = {
            "token_loss": "loss_token",
            "coord_loss": "loss_coord",
            "dist_loss": "loss_dist",
            "x_norm_loss": "loss_x_norm",
            "delta_pair_norm_loss": "loss_delta_pair_norm",
        }
        for mine, ref in expected.items():
            with self.subTest(term=mine):
                np.testing.assert_allclose(
                    float(more[mine].detach()),
                    float(self.golden[f"small_fp64/loss/{ref}"]),
                    rtol=1e-5,
                    atol=1e-8,
                )
        np.testing.assert_allclose(
            float(total.detach()),
            float(self.golden["small_fp64/loss/loss_total"]),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_gradients_flow(self) -> None:
        coord, atype, nlist = self.inputs()
        model = self.build_torch_model(coord.shape[1])
        ret = model.forward_lower(
            torch.as_tensor(coord.reshape(len(self.n_real), -1)),
            torch.as_tensor(atype),
            torch.as_tensor(nlist),
            None,
        )
        ret["token_logits"].sum().backward()
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(bool(torch.any(g != 0)) for g in grads))


if __name__ == "__main__":
    unittest.main()
