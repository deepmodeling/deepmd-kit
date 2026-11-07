# SPDX-License-Identifier: LGPL-3.0-or-later
"""The reference result channel is independent of native initialization logs."""

import pytest
from python_reference import (
    run_python_reference,
)


@pytest.mark.parametrize("result", [{"energy": [1.0, -2.0]}, [1.0, {"force": 0.0}]])
def test_reference_ignores_diagnostic_output(result: object) -> None:
    script = (
        "import os\n"
        "print('Warp initialized')\n"
        "os.write(1, b'native stdout diagnostic\\n')\n"
        "os.write(2, b'native stderr diagnostic\\n')\n"
        f"result = {result!r}\n"
    )
    assert run_python_reference(script) == result


def test_reference_failure_preserves_diagnostics() -> None:
    with pytest.raises(RuntimeError, match="reference failed") as error:
        run_python_reference(
            "print('initialization diagnostic')\n"
            "raise RuntimeError('reference failed')\n"
        )
    assert "initialization diagnostic" in str(error.value)
