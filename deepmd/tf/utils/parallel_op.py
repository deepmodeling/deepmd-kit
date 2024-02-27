# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Tuple,
)

from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.sess import (
    run_sess,
)


class ParallelOp:
    """Run an op with data parallelism.

    Parameters
    ----------
    builder : Callable[..., Tuple[Dict[str, tf.Tensor], Tuple[tf.Tensor]]]
        returns two objects: a dict which stores placeholders by key, and a tuple with the final op(s)
    nthreads : int, optional
        the number of threads
    config : tf.ConfigProto, optional
        tf.ConfigProto

    Examples
    --------
    >>> from deepmd.tf.env import tf
    >>> from deepmd.tf.utils.parallel_op import ParallelOp
    >>> def builder():
    ...     x = tf.placeholder(tf.int32, [1])
    ...     return {"x": x}, (x + 1)
    >>> p = ParallelOp(builder, nthreads=4)
    >>> def feed():
    ...     for ii in range(10):
    ...         yield {"x": [ii]}
    >>> print(*p.generate(tf.Session(), feed()))
    [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
    """

    def __init__(
        self,
        builder: Callable[..., Tuple[Dict[str, tf.Tensor], Tuple[tf.Tensor]]],
        nthreads: Optional[int] = None,
        config: Optional[tf.ConfigProto] = None,
    ) -> None:
        if nthreads is not None:
            self.nthreads = nthreads
        elif config is not None:
            self.nthreads = max(config.inter_op_parallelism_threads, 1)
        else:
            self.nthreads = 1

        self.placeholders = []
        self.ops = []
        for ii in range(self.nthreads):
            with tf.name_scope("task_%d" % ii) as scope:
                placeholder, op = builder()
                self.placeholders.append(placeholder)
                self.ops.append(op)

    def generate(
        self, sess: tf.Session, feed: Generator[Dict[str, Any], None, None]
    ) -> Generator[Tuple, None, None]:
        """Returns a generator.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow session
        feed : Generator[dict, None, None]
            generator which yields feed_dict

        Yields
        ------
        Generator[Tuple, None, None]
            generator which yields session returns
        """
        nn = self.nthreads
        while True:
            feed_dict = {}
            for ii in range(self.nthreads):
                try:
                    fd = next(feed)
                except StopIteration:
                    if ii == 0:
                        return
                    nn = ii
                    break
                for kk, vv in fd.items():
                    feed_dict[self.placeholders[ii][kk]] = vv
            ops = self.ops[:nn]
            yield from run_sess(sess, ops, feed_dict=feed_dict)
