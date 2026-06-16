# SPDX-License-Identifier: LGPL-3.0-or-later
"""AOTInductor settings for fast test-only ``.pt2`` compilation."""


def apply_fast_aoti_compile_config() -> None:
    """Reduce AOTInductor compile time for test fixtures.

    The test suite validates correctness rather than runtime performance, so it
    can skip expensive Inductor/C++ optimization passes while producing usable
    ``.pt2`` artifacts for Python, C, C++, and LAMMPS tests.
    """
    import torch._inductor.config as inductor_config

    inductor_config.max_fusion_size = 8
    inductor_config.epilogue_fusion = False
    inductor_config.pattern_matcher = False
    inductor_config.aot_inductor.package_cpp_only = True
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
