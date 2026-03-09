# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for dpmodel spin model finetune: _get_spin_sampled_func, change_out_bias, change_type_map."""

import copy
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.spin_model import (
    SpinModel,
)
from deepmd.utils.spin import (
    Spin,
)

from ...seed import (
    GLOBAL_SEED,
)


def _make_spin_model(
    type_map: list[str],
    use_spin: list[bool],
    virtual_scale: list[float],
    sel: list[int],
    rcut: float = 4.0,
    rcut_smth: float = 0.5,
    numb_fparam: int = 0,
    default_fparam: list[float] | None = None,
) -> SpinModel:
    """Create a dpmodel SpinModel for testing."""
    # The backbone model sees both real and virtual types
    ntypes_real = len(type_map)
    type_map_backbone = type_map + [t + "_spin" for t in type_map]
    # sel needs to be doubled for virtual types
    sel_backbone = sel + sel

    descriptor = DescrptSeA(
        rcut=rcut,
        rcut_smth=rcut_smth,
        sel=sel_backbone,
        seed=GLOBAL_SEED,
    )
    fitting = InvarFitting(
        "energy",
        ntypes_real * 2,  # backbone sees real + virtual types
        descriptor.get_dim_out(),
        1,
        mixed_types=descriptor.mixed_types(),
        seed=GLOBAL_SEED,
        numb_fparam=numb_fparam,
        default_fparam=default_fparam,
    )

    from deepmd.dpmodel.atomic_model.dp_atomic_model import (
        DPAtomicModel,
    )
    from deepmd.dpmodel.common import (
        NativeOP,
    )
    from deepmd.dpmodel.model.base_model import (
        BaseModel,
    )
    from deepmd.dpmodel.model.make_model import (
        make_model,
    )

    CM = make_model(DPAtomicModel, T_Bases=(NativeOP, BaseModel))
    backbone = CM(descriptor, fitting, type_map=type_map_backbone)

    spin = Spin(use_spin=use_spin, virtual_scale=virtual_scale)
    return SpinModel(backbone_model=backbone, spin=spin)


def _make_sample_data(
    nframes: int,
    nloc: int,
    ntypes: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """Create fake sample data for testing."""
    atype = rng.randint(0, ntypes, size=(nframes, nloc)).astype(np.int64)
    coord = rng.randn(nframes, nloc, 3).astype(np.float64)
    spin = 0.5 * rng.randn(nframes, nloc, 3).astype(np.float64)
    energy = rng.randn(nframes, 1).astype(np.float64)
    natoms_count = np.zeros((nframes, 2 + ntypes), dtype=np.int32)
    natoms_count[:, 0] = nloc
    natoms_count[:, 1] = nloc
    for i in range(nframes):
        for t in range(ntypes):
            natoms_count[i, 2 + t] = np.sum(atype[i] == t)
    return [
        {
            "coord": coord,
            "atype": atype,
            "spin": spin,
            "energy": energy,
            "natoms": natoms_count,
            "find_energy": np.float32(1.0),
            "find_fparam": np.float32(0.0),
        }
    ]


class TestSpinModelGetSpinSampledFunc(unittest.TestCase):
    """Test _get_spin_sampled_func correctly transforms spin data (dpmodel)."""

    def setUp(self) -> None:
        self.type_map = ["Ni", "O"]
        self.model = _make_spin_model(
            type_map=self.type_map,
            use_spin=[True, False],
            virtual_scale=[0.3140],
            sel=[10, 10],
        )
        self.rng = np.random.RandomState(GLOBAL_SEED)

    def test_spin_data_transformation(self) -> None:
        nframes, nloc, ntypes = 2, 6, 2
        sampled = _make_sample_data(nframes, nloc, ntypes, self.rng)

        def sampled_func() -> list[dict]:
            return sampled

        spin_sampled_func = self.model._get_spin_sampled_func(sampled_func)
        spin_sampled = spin_sampled_func()

        for i, sys_data in enumerate(spin_sampled):
            original = sampled[i]
            # coord should be doubled (real + virtual)
            assert sys_data["coord"].shape[1] == 2 * nloc
            # atype should be doubled
            assert sys_data["atype"].shape[1] == 2 * nloc
            # spin should not be in the transformed data
            assert "spin" not in sys_data
            # energy should be preserved
            np.testing.assert_array_equal(sys_data["energy"], original["energy"])
            # natoms should be transformed correctly
            natoms = original["natoms"]
            expected_natoms = np.concatenate(
                [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], axis=-1
            )
            np.testing.assert_array_equal(sys_data["natoms"], expected_natoms)

    def test_coord_values(self) -> None:
        """Verify virtual coordinates = real + spin * virtual_scale."""
        nframes, nloc, ntypes = 1, 4, 2
        sampled = _make_sample_data(nframes, nloc, ntypes, self.rng)

        spin_sampled = self.model._get_spin_sampled_func(lambda: sampled)()

        original = sampled[0]
        transformed = spin_sampled[0]
        coord = original["coord"]  # (nframes, nloc, 3)
        spin = original["spin"]
        atype = original["atype"]
        virtual_scale_mask = self.model.virtual_scale_mask

        # Real coords should be unchanged
        np.testing.assert_array_equal(transformed["coord"][:, :nloc], coord)
        # Virtual coords = real + spin * scale
        expected_virtual = coord + spin * virtual_scale_mask[atype].reshape(
            nframes, nloc, 1
        )
        np.testing.assert_allclose(
            transformed["coord"][:, nloc:], expected_virtual, atol=1e-12
        )


class TestSpinModelChangeOutBias(unittest.TestCase):
    """Test change_out_bias for dpmodel SpinModel."""

    def setUp(self) -> None:
        self.type_map = ["Ni", "O"]
        self.model = _make_spin_model(
            type_map=self.type_map,
            use_spin=[True, False],
            virtual_scale=[0.3140],
            sel=[10, 10],
        )
        self.rng = np.random.RandomState(GLOBAL_SEED)

    def test_change_out_bias_runs(self) -> None:
        """Test that change_out_bias does not raise with spin model."""
        sampled = _make_sample_data(2, 6, 2, self.rng)
        old_bias = copy.deepcopy(self.model.backbone_model.get_out_bias())
        self.model.change_out_bias(sampled, bias_adjust_mode="set-by-statistic")
        new_bias = self.model.backbone_model.get_out_bias()
        # Bias should have changed
        assert not np.allclose(old_bias, new_bias), "Bias was not updated"

    def test_change_out_bias_with_callable(self) -> None:
        """Test change_out_bias with a callable (lazy sampled func)."""
        sampled = _make_sample_data(2, 6, 2, self.rng)
        old_bias = copy.deepcopy(self.model.backbone_model.get_out_bias())
        self.model.change_out_bias(lambda: sampled, bias_adjust_mode="set-by-statistic")
        new_bias = self.model.backbone_model.get_out_bias()
        assert not np.allclose(old_bias, new_bias), "Bias was not updated"


class TestSpinModelChangeTypeMap(unittest.TestCase):
    """Test change_type_map for dpmodel SpinModel."""

    def setUp(self) -> None:
        self.type_map = ["Ni", "O"]
        self.model = _make_spin_model(
            type_map=self.type_map,
            use_spin=[True, False],
            virtual_scale=[0.3140],
            sel=[10, 10],
        )

    def test_change_type_map(self) -> None:
        """Test that change_type_map delegates to backbone with _spin suffixes.

        DescrptSeA does not support change_type_map, so we verify that the
        SpinModel correctly constructs the suffixed type map and delegates
        to the backbone (which raises NotImplementedError from se_e2_a).
        The full change_type_map workflow is tested via PT backend with
        mixed-types descriptors.
        """
        new_type_map = ["O", "Ni"]
        with self.assertRaises(NotImplementedError):
            self.model.change_type_map(new_type_map)


class TestSpinModelWithDefaultFparam(unittest.TestCase):
    """Test _get_spin_sampled_func injects default fparam (dpmodel)."""

    def setUp(self) -> None:
        self.type_map = ["Ni", "O"]
        self.default_fparam = [0.5, 1.0]
        self.model = _make_spin_model(
            type_map=self.type_map,
            use_spin=[True, False],
            virtual_scale=[0.3140],
            sel=[10, 10],
            numb_fparam=2,
            default_fparam=self.default_fparam,
        )
        self.rng = np.random.RandomState(GLOBAL_SEED)

    def test_fparam_injected(self) -> None:
        """Test that _get_spin_sampled_func + _make_wrapped_sampler injects fparam."""
        sampled = _make_sample_data(2, 6, 2, self.rng)
        assert "fparam" not in sampled[0]

        spin_sampled = self.model._get_spin_sampled_func(lambda: sampled)()

        for sys_data in spin_sampled:
            assert "fparam" in sys_data, (
                "_make_wrapped_sampler did not inject default fparam"
            )
            nframe = sys_data["atype"].shape[0]
            assert sys_data["fparam"].shape == (nframe, 2)
            np.testing.assert_allclose(sys_data["fparam"][0], self.default_fparam)


if __name__ == "__main__":
    unittest.main()
