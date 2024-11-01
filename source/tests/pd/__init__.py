# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle

paddle.framework.core.set_num_threads(1)
# paddle.set_num_interop_threads(1)
# testing purposes; device should always be set explicitly
# paddle.set_device("gpu:9999999")
