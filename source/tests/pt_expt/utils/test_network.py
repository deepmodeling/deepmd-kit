# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.pt_expt.utils.network import (
    NativeLayer,
)


def test_native_layer_clears_parameter_on_none() -> None:
    layer = NativeLayer(2, 3, trainable=True)
    assert layer.w is not None
    layer.w = None
    assert layer.w is None
    assert layer._parameters.get("w") is None


def test_native_layer_clears_buffer_on_none() -> None:
    layer = NativeLayer(2, 3, trainable=False)
    assert layer.w is not None
    layer.w = None
    assert layer.w is None
    assert layer._buffers.get("w") is None
