# SPDX-License-Identifier: LGPL-3.0-or-later

import os

if os.environ.get("DP_CI_IMPORT_PADDLE_BEFORE_TF", "0") == "1":
    # Paddle must be loaded before TensorFlow in the CI test process.
    import paddle  # noqa: F401
    import tensorflow  # noqa: F401
