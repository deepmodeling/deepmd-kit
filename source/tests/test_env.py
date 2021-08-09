import unittest

from deepmd import env
from unittest import mock


class TestTFThreadCount(unittest.TestCase):
    @mock.patch.dict('os.environ', values={})
    def test_empty(self):
        intra, inter = env.get_tf_default_nthreads()
        self.assertEqual(intra, 0)
        self.assertEqual(inter, 0)

    @mock.patch.dict('os.environ', values={
        'TF_INTRA_OP_PARALLELISM_THREADS': '5',
        'TF_INTER_OP_PARALLELISM_THREADS': '3'
    })
    def test_given(self):
        intra, inter = env.get_tf_default_nthreads()
        self.assertEqual(intra, 5)
        self.assertEqual(inter, 3)


class TestTFSessionConfig(unittest.TestCase):
    def test_default(self):
        shared = env.default_tf_session_config
        new = env.get_tf_session_config()
        self.assertNotEqual(id(shared), id(new))

    @mock.patch('deepmd.env.get_tf_default_nthreads')
    def test_get(self, mock_method):
        mock_method.return_value = (5, 3)
        config = env.get_tf_session_config()
        self.assertEqual(config.intra_op_parallelism_threads, 5)
        self.assertEqual(config.inter_op_parallelism_threads, 3)

    def test_reset(self):
        shared = env.default_tf_session_config
        env.reset_default_tf_session_config(True)
        self.assertEqual(shared.device_count['GPU'], 0)
        env.reset_default_tf_session_config(False)
        self.assertEqual(len(shared.device_count), 0)
