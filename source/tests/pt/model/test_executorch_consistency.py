# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from contextlib import (
    contextmanager,
)

import torch
import torch.export
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
)
from executorch.runtime import (
    Runtime,
)

from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


@contextmanager
def _cpu_default_device():
    prior_device = torch._C._get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(prior_device)


class TestExecutorchConsistency(unittest.TestCase):
    def setUp(self):
        self.rcut = 6.0
        self.rcut_smth = 5.0
        self.sel = [4, 4]
        self.ntypes = 2
        self.neuron = [10, 10]
        self.axis_neuron = 4
        self.precision = "float32"  # Executorch primarily supports float32
        self.dtype = torch.float32

    def _test_descriptor_consistency(self, model, coord_ext, atype_ext, nlist):
        model.eval()

        # 1. Run pristine model
        with torch.no_grad():
            expected_output = model(coord_ext, atype_ext, nlist)
            # The descriptor returns a tuple, usually the first element is the main descriptor
            if isinstance(expected_output, tuple):
                expected_output = expected_output[0]

        # 2. Export and compile to Executorch
        try:
            exported_program = torch.export.export(model, (coord_ext, atype_ext, nlist))
            with _cpu_default_device():
                edge_program = to_edge(
                    exported_program,
                    compile_config=EdgeCompileConfig(
                        _core_aten_ops_exception_list=[torch.ops.aten.sort.stable]
                    ),
                )
            executorch_program = edge_program.to_executorch()
        except Exception as e:
            self.fail(f"Executorch compilation failed for {type(model).__name__}: {e}")

        # 3. Execute with Executorch Runtime
        program = Runtime.get().load_program(executorch_program.buffer)

        # Prepare inputs for executorch
        # Executorch runtime inputs usually need to be flat list of tensors or similar depending on the API
        # The current python API for runtime might vary slightly by version.

        # Using the low-level API style or the higher level if available.
        # Assuming `forward` method is the entry point (index 0 usually).

        # Load the method (plan)
        method_name = "forward"
        try:
            # Prepare inputs
            # Note: Executorch runtime expects inputs to be compatible with the memory plan
            # Here we simply pass the tensors.
            # The execution model in python bindings typically takes a list of inputs.
            inputs = [coord_ext, atype_ext, nlist]

            # Execute
            # Note: The specific API call might be `run`, `execute`, or similar.
            # Based on standard usage pattern:
            method = program.load_method(method_name)
            result = method.execute(inputs)

            # The result is typically a list of outputs
            actual_output = result[0]

            # 4. Compare results
            # Allow for some tolerance due to different backends/precisions if any
            torch.testing.assert_close(
                actual_output, expected_output, rtol=1e-4, atol=1e-4
            )

        except Exception as e:
            self.fail(f"Executorch execution failed: {e}")

    def test_se_e2_a_consistency(self):
        model = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.neuron,
            axis_neuron=self.axis_neuron,
            precision=self.precision,
            trainable=False,
        ).to(env.DEVICE)

        nf = 1
        nloc = 5
        coord = torch.randn(nf, nloc * 3, device=env.DEVICE, dtype=self.dtype)
        atype = torch.randint(
            0, self.ntypes, (nf, nloc), dtype=torch.int32, device=env.DEVICE
        )
        coord_ext, atype_ext, _, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.rcut,
            self.sel,
            mixed_types=model.mixed_types(),
        )
        coord_ext = coord_ext.view(nf, -1)

        self._test_descriptor_consistency(model, coord_ext, atype_ext, nlist)

    def test_dpa1_consistency(self):
        # Note: DPA1 failed compilation in previous turn due to var_mean.correction
        # We include it here. If it fails compilation, the helper returns early or we can catch it.
        # Ideally we fix the compilation or skip if known broken.
        model = DescrptDPA1(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            ntypes=self.ntypes,
            neuron=self.neuron,
            axis_neuron=self.axis_neuron,
            precision=self.precision,
            trainable=False,
        ).to(env.DEVICE)

        nf = 1
        nloc = 5
        coord = torch.randn(nf, nloc * 3, device=env.DEVICE, dtype=self.dtype)
        atype = torch.randint(
            0, self.ntypes, (nf, nloc), dtype=torch.int32, device=env.DEVICE
        )
        coord_ext, atype_ext, _, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.rcut,
            self.sel,
            mixed_types=model.mixed_types(),
        )
        coord_ext = coord_ext.view(nf, -1)

        self._test_descriptor_consistency(model, coord_ext, atype_ext, nlist)

    def test_full_model_consistency(self):
        # Full EnergyModel with se_e2_a descriptor
        model_params = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": self.sel,
                "rcut_smth": self.rcut_smth,
                "rcut": self.rcut,
                "neuron": self.neuron,
                "axis_neuron": self.axis_neuron,
                "precision": self.precision,
            },
            "fitting_net": {
                "type": "direct_force_ener",
                "neuron": [10, 10],
                "precision": self.precision,
            },
        }
        model = get_model(model_params).to(env.DEVICE)
        model.eval()

        nf = 1
        nloc = 5
        coord = torch.randn(nf, nloc * 3, device=env.DEVICE, dtype=self.dtype)
        atype = torch.randint(
            0, self.ntypes, (nf, nloc), dtype=torch.int32, device=env.DEVICE
        )
        coord_ext, atype_ext, _, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.rcut,
            self.sel,
            mixed_types=model.mixed_types(),
        )
        coord_ext = coord_ext.view(nf, -1)

        # 1. Run pristine model (forward_lower)
        # Note: forward_lower returns a dict. Executorch output will be a flat tuple of tensors (values of the dict usually, or based on graph return).
        # We need to wrap it to return specific tensor(s) or handle dict if export supports it (export usually flattens).

        # Pristine output (dict)
        pristine_dict = model.forward_lower(coord_ext, atype_ext, nlist)
        # We'll focus on 'energy' and 'force' (if available) for consistency
        # force requires grad calculation which we fixed.

        class ForwardLowerWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, extended_coord, extended_atype, nlist):
                ret = self.model.forward_lower(extended_coord, extended_atype, nlist)
                # Return tuple of values to match typical export behavior for dicts or flatten them
                # Usually we want energy and force
                # We can return the values of the dict. Torch export might sort them or we define order.
                # Let's return explicit keys we care about.
                return ret["energy"], ret["atom_energy"], ret["dforce"]

        wrapper = ForwardLowerWrapper(model)

        # Recalculate pristine with wrapper to be sure
        # Note: We cannot use torch.no_grad() here because the model calculates forces using autograd.grad,
        # which requires the computation graph of energy to be active.
        expected_energy, expected_atom_energy, expected_force = wrapper(
            coord_ext, atype_ext, nlist
        )

        # 2. Export and compile
        try:
            # Torch export may lift tensor constants created inside the model into the
            # graph. Some are produced in fake mode and can trigger strict failures.
            # Relax this check for the full-model export to keep the consistency test
            # focused on executorch compatibility.
            with torch._export.config.patch(error_on_lifted_constant_tensors=False):
                exported_program = torch.export.export(
                    wrapper, (coord_ext, atype_ext, nlist)
                )
            with _cpu_default_device():
                edge_program = to_edge(
                    exported_program,
                    compile_config=EdgeCompileConfig(
                        _core_aten_ops_exception_list=[torch.ops.aten.sort.stable]
                    ),
                )
            executorch_program = edge_program.to_executorch()
        except Exception as e:
            self.fail(f"Full model compilation failed: {e}")

        # 3. Execute
        program = Runtime.get().load_program(executorch_program.buffer)
        inputs = [coord_ext, atype_ext, nlist]

        try:
            method = program.load_method("forward")
            result = method.execute(inputs)
            # Result should be list of [energy, atom_energy, dforce]
            actual_energy = result[0]
            actual_atom_energy = result[1]
            actual_force = result[2]

            # 4. Compare
            torch.testing.assert_close(
                actual_energy,
                expected_energy,
                rtol=1e-4,
                atol=1e-4,
                msg="Energy mismatch",
            )
            torch.testing.assert_close(
                actual_atom_energy,
                expected_atom_energy,
                rtol=1e-4,
                atol=1e-4,
                msg="Atom energy mismatch",
            )
            torch.testing.assert_close(
                actual_force, expected_force, rtol=1e-4, atol=1e-4, msg="Force mismatch"
            )

        except Exception as e:
            self.fail(f"Full model execution failed: {e}")


if __name__ == "__main__":
    unittest.main()
