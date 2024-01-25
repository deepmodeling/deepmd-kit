# SPDX-License-Identifier: LGPL-3.0-or-later
import os

from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.errors import (
    OutOfMemoryError,
)


def run_sess(sess: tf.Session, *args, **kwargs):
    """Run session with erorrs caught.

    Parameters
    ----------
    sess : tf.Session
        TensorFlow Session
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Any
        the result of sess.run()
    """
    try:
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#run
        return sess.run(*args, **kwargs)
    except (tf.errors.ResourceExhaustedError, tf.errors.CancelledError) as e:
        MESSAGE = (
            "Your memory may be not enough, thus an error has been raised "
            "above. You need to take the following actions:\n"
            "1. Check if the network size of the model is too large.\n"
            "2. Check if the batch size of training or testing is too large."
            " You can set the training batch size to `auto`.\n"
            "3. Check if the number of atoms is too large.\n"
        )
        if tf.test.is_built_with_cuda():
            MESSAGE += (
                "4. Check if another program is using the same GPU by "
                "execuating `nvidia-smi`. The usage of GPUs is "
                "controlled by `CUDA_VISIBLE_DEVICES` environment "
                "variable (current value: {}).\n".format(
                    os.getenv("CUDA_VISIBLE_DEVICES", None)
                )
            )
        raise OutOfMemoryError(MESSAGE) from e
