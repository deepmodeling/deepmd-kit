import unittest

from deepmd.cluster import local, slurm
from unittest import mock


kHostName = 'org.deepmd.unittest'


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
            gpus = local.get_gpus()
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
        'SLURM_JOB_NODELIST': 'org.deepmd.host-[3-5],com.github.jack',
        'SLURMD_NODENAME': 'org.deepmd.host-4',
        'SLURM_JOB_NUM_NODES': '4'
    })
    def test_multiple(self):
        nodename, nodelist, _ = slurm.get_resource()
        self.assertEqual(nodename, 'org.deepmd.host-4')
        self.assertEqual(nodelist, [
            'org.deepmd.host-3',
            'org.deepmd.host-4',
            'org.deepmd.host-5',
            'com.github.jack'
        ])

    def test_illegal(self):
        environ = {
            'SLURM_JOB_NODELIST': 'org.deepmd.host-[3-5]',
            'SLURMD_NODENAME': 'org.deepmd.host-4'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(RuntimeError) as cm:
                nodename, nodelist, _ = slurm.get_resource()
                self.assertIn('Could not get SLURM number', str(cm.exception))

        environ = {
            'SLURM_JOB_NODELIST': 'org.deepmd.mike,com.github.jack',
            'SLURMD_NODENAME': 'org.deepmd.mike',
            'SLURM_JOB_NUM_NODES': '4'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(ValueError) as cm:
                nodename, nodelist, _ = slurm.get_resource()
                self.assertIn('Number of slurm nodes 2', str(cm.exception))

        environ = {
            'SLURM_JOB_NODELIST': 'org.deepmd.bob,com.github.jack',
            'SLURMD_NODENAME': 'org.deepmd.mike',
            'SLURM_JOB_NUM_NODES': '2'
        }
        with mock.patch.dict('os.environ', environ):
            with self.assertRaises(ValueError) as cm:
                nodename, nodelist, _ = slurm.get_resource()
                self.assertIn('Nodename(org.deepmd.mike', str(cm.exception))
