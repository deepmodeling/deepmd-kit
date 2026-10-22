# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch extras keep Vesin bindings within their supported versions."""

import pytest
from packaging.requirements import (
    Requirement,
)

from backend.find_pytorch import (
    get_pt_requirement,
)


@pytest.mark.parametrize("version", ["2.1.0", "2.2.0", "2.3.0", ""])
def test_vesin_requirements(version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_VERSION", raising=False)
    monkeypatch.setenv("CIBUILDWHEEL", "0")
    get_pt_requirement.cache_clear()
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, get_pt_requirement(version)["torch"])
    }
    get_pt_requirement.cache_clear()
    if version in ("2.1.0", "2.2.0"):
        assert "vesin" not in requirements
        assert "vesin-torch" not in requirements
        return

    assert requirements["vesin"].extras == {"torch"}
    for name in ("vesin", "vesin-torch"):
        requirement = requirements[name]
        assert "0.5.9" not in requirement.specifier
        assert "0.6" in requirement.specifier
        for system, machine, supported in (
            ("Darwin", "x86_64", False),
            ("Darwin", "arm64", True),
            ("Linux", "x86_64", True),
            ("Windows", "AMD64", True),
        ):
            assert (
                requirement.marker.evaluate(
                    {"platform_system": system, "platform_machine": machine}
                )
                is supported
            )
