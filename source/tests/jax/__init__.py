# SPDX-License-Identifier: LGPL-3.0-or-later
import os

os.environ["XLA_FLAGS"] = " ".join(
    (
        "--xla_cpu_multi_thread_eigen=false",
        "intra_op_parallelism_threads=1",
        "inter_op_parallelism_threads=1",
    )
)
