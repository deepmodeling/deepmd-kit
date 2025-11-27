# SPDX-License-Identifier: LGPL-3.0-or-later
import os


def main():
    current_dir = os.path.abspath(os.getcwd())
    script_dir = os.path.abspath(os.path.dirname(__file__))

    if current_dir != script_dir:
        raise RuntimeError(
            f"[ERROR] Please run this script under directory: `{script_dir}`"
        )

    from paddle.utils.cpp_extension import (
        CppExtension,
        setup,
    )

    setup(name="deepmd_op_pd", ext_modules=CppExtension(sources=["comm.cc"]))


if __name__ == "__main__":
    main()
