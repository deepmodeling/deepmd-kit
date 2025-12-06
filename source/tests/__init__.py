# SPDX-License-Identifier: LGPL-3.0-or-later

import os

if os.environ.get("DP_CI_IMPORT_PADDLE_BEFORE_TF", "0") == "1":
    import paddle as _
    import tensorflow as _  # noqa: F401
