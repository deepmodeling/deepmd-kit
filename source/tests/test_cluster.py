import unittest

from deepmd.cluster import local, slurm
from unittest import mock


kHostName = 'compute-b24-1'


class FakePopen(object):
    def __init__(self, stdout=b'', stderr=b'', returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode

    def communicate(self):
        return self._stdout, self._stderr

    @property
    def returncode(self):
        return self._returncode


class TestGPU(unittest.TestCase):
    @mock.patch('subprocess.Popen')
    def test_none(self, mock_Popen):
        mock_Popen.return_value.__enter__.return_value = FakePopen(b'0', b'')
        gpus = local.get_gpus()
        self.assertIsNone(gpus)

    @mock.patch('subprocess.Popen')
    def test_valid(self, mock_Popen):
        mock_Popen.return_value.__enter__.return_value = FakePopen(b'2', b'')
        gpus = local.get_gpus()
        self.assertEqual(gpus, [0, 1])

    @mock.patch('subprocess.Popen')
    def test_error(self, mock_Popen):
        mock_Popen.return_value.__enter__.return_value = \
            FakePopen(stderr=b'!', returncode=1)
        with self.assertRaises(RuntimeError) as cm:
            _ = local.get_gpus()
            self.assertIn('Failed to detect', str(cm.exception))


class TestLocal(unittest.TestCase):
    @mock.patch('socket.gethostname')
    def test_resource(self, mock_gethostname):
        mock_gethostname.return_value = kHostName
        nodename, nodelist, _ = local.get_resource()
        self.assertEqual(nodename, kHostName)
        self.assertEqual(nodelist, [kHostName])


class TestSlurm(unittest.TestCase):
    @mock.patch.dict('os.environ', values={
        'SLURM_JOB_NODELIST': kHostName,
        'SLURMD_NODENAME': kHostName,
        'SLURM_JOB_NUM_NODES': '1'
    })
    def test_single(self):
        nodename, nodelist, _ = slurm.get_resource()
        self.assertEqual(nodename, kHostName)
        self.assertEqual(nodelist, [kHostName])

    @mock.patch.dict('os.environ', values={
        'SLURM_JOB_NODELIST': 'compute-b24-[1-3,5-9],compute-b25-[4,8]',
        'SLURMD_NODENAME': 'compute-b24-2',
        'SLURM_JOB_NUM_NODES': '10'
    })
    def test_multiple(self):
        nodename, nodelist, _ = slurm.get_resource()
        self.assertEqual(nodename, 'compute-b24-2')
        self.assertEqual(nodelist, [
            'compute-b24-1',
            'compute-b24-2',
            'compute-b24-3',
            'compute-b24-5',
            'compute-b24-6',
            'compute-b24-7',
            'compute-b24-8',
            'compute-b24-9',
            'compute-b25-4',
            'compute-b25-8'
        ])

    def test_illegal(self):
        environ = {
            'SLURM_JOB_NODELIST': 'compute-b24-[3-5]',
            'SLURMD_NODENAME': 'compute-b24-4'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(RuntimeError) as cm:
                _ = slurm.get_resource()
                self.assertIn('Could not get SLURM number', str(cm.exception))

        environ = {
            'SLURM_JOB_NODELIST': 'compute-b24-1,compute-b25-2',
            'SLURMD_NODENAME': 'compute-b25-2',
            'SLURM_JOB_NUM_NODES': '4'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(ValueError) as cm:
                _ = slurm.get_resource()
                self.assertIn('Number of slurm nodes 2', str(cm.exception))

        environ = {
            'SLURM_JOB_NODELIST': 'compute-b24-1,compute-b25-3',
            'SLURMD_NODENAME': 'compute-b25-2',
            'SLURM_JOB_NUM_NODES': '2'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(ValueError) as cm:
                _ = slurm.get_resource()
                self.assertIn('Nodename(compute-b25-2', str(cm.exception))
