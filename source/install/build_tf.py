#!/usr/bin/env python3
"""The easy script to build TensorFlow C++ Library.

Required dependencies:
- gcc/g++
- Python3
- NumPy
- git
For CUDA only:
- CUDA Toolkit
- cuDNN
"""

# make sure Python 3 is used
# https://stackoverflow.com/a/41901923/9567349
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

# The script should only rely on the stardard Python libraries.

from contextlib import contextmanager
import argparse
import importlib.util
import os
import re
import stat
import subprocess as sp
import hashlib
import logging
import urllib.request
import tarfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import lru_cache
from shutil import copytree, ignore_patterns, copy2
from fnmatch import filter


# default config
FILE = Path(__file__).parent.absolute()
PACKAGE_DIR = FILE.parent / "packages"
PREFIX = None
CPU_COUNT = os.cpu_count()
nvcc_path = shutil.which("nvcc")
if nvcc_path is not None:
    CUDA_PATH = Path(shutil.which("nvcc")).parent.parent
else:
    CUDA_PATH = None
CUDNN_PATH = Path("/usr") if os.path.isfile("/usr/include/cudnn.h") else None
GCC = shutil.which("gcc")
GXX = shutil.which("g++")


dlog = logging.getLogger("TensorFlow C++ Library installer")
dlog.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
dlog.addHandler(handler)


# Common utils

def download_file(url: str, filename: str):
    """Download files from remote URL.

    Parameters
    ----------
    url: str
        The URL that is available to download.
    filename: str
        The downloading path of the file.

    Raises
    ------
    URLError
        raises for HTTP error
    """
    dlog.info("Download %s from %s" % (filename, url))
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


class OnlineResource:
    """Online resource. Call the instance to download.

    Parameters
    ----------
    filename: str
        The target filename.
    url : str
        remote URL
    sha256 : str
        expecting sha256
    executable : bool, default=False
        if the file is executable
    gzip : str
        if not None, decompress to a directory
    """

    def __init__(self,
                 filename: str,
                 url: str,
                 sha256: str = None,
                 executable: bool = False,
                 gzip: str = None,
                 ) -> None:
        self.filename = filename
        self.url = url
        self.reference_sha256 = sha256
        self.executable = executable
        self.gzip = gzip

    def __call__(self):
        # download if not exists
        if not self.exists:
            self.download()
            if not self.exists:
                raise RuntimeError(
                    "Download {} from {} failed! "
                    "You can manually download it to {} and "
                    "retry the script.".format(
                        self.filename, self.url, str(self.path)
                    ))
        self.post_process()

    def post_process(self):
        if self.executable:
            self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC)
        if self.gzip is not None:
            with tarfile.open(self.path) as tar:
                tar.extractall(path=self.gzip_path)

    def download(self):
        """Download the target file."""
        download_file(self.url, self.path)

    @property
    def path(self) -> Path:
        """Path to the target file."""
        return PACKAGE_DIR / self.filename

    @property
    def gzip_path(self) -> Path:
        if self.gzip is None:
            raise RuntimeError("gzip is None for %s" % self.path)
        return PACKAGE_DIR / self.gzip

    @property
    def sha256(self) -> str:
        """Get sha256 of the target file.

        Parameters
        ----------
        filename : str
            The filename.

        Returns
        -------
        sha256 : str
            The sha256.
        """
        h = hashlib.sha256()
        # buffer size: 128 kB
        b = bytearray(128*1024)
        mv = memoryview(b)
        with open(self.path, 'rb', buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        return h.hexdigest()

    @property
    def exists(self) -> bool:
        """Check if target file exists."""
        return self.path.exists() and (self.sha256 == self.reference_sha256 or self.reference_sha256 is None)


class Build(metaclass=ABCMeta):
    """Build process."""
    @abstractproperty
    def resources(self) -> Dict[str, OnlineResource]:
        """Required resources."""

    @abstractproperty
    def dependencies(self) -> Dict[str, "Build"]:
        """Required dependencies."""

    def download_all_resources(self):
        """All resources, including dependencies' resources."""
        for res in self.resources.values():
            res()
        for dd in self.dependencies.values():
            if not dd.built:
                dd.download_all_resources()

    @abstractproperty
    def built(self) -> bool:
        """Check if it has built."""

    @abstractmethod
    def build(self):
        """Build process."""

    def __call__(self):
        if not self.built:
            # firstly download all resources
            self.download_all_resources()
            for dd in self.dependencies.values():
                if not dd.built:
                    dd()
                else:
                    dlog.info("Skip installing %s, which has been already installed" % dd.__class__.__name__)
            dlog.info("Start installing %s..." % self.__class__.__name__)
            with tempfile.TemporaryDirectory() as tmpdirname:
                self._prefix = Path(tmpdirname)
                self.build()
                self.copy_from_tmp_to_prefix()
            if not self.built:
                raise RuntimeError("Build failed!")

    @property
    def prefix(self):
        """Tmp prefix"""
        return self._prefix

    def copy_from_tmp_to_prefix(self):
        """Copy from tmp prefix to real prefix."""
        copytree2(str(self.prefix), str(PREFIX))


@contextmanager
def set_directory(path: Path):
    """Sets the current working path within the context.

    Parameters
    ----------
    path : Path
        The path to the cwd

    Yields
    ------
    None

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    cwd = Path().absolute()
    path.mkdir(exist_ok=True, parents=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def list2env(l: list) -> str:
    return ':'.join(map(str, l))


def get_shlib_ext():
    """Return the shared library extension."""
    plat = sys.platform
    if plat.startswith('win'):
        return '.dll'
    elif plat in ['osx', 'darwin']:
        return '.dylib'
    elif plat.startswith('linux'):
        return '.so'
    else:
        raise NotImplementedError(plat)


def copy3(src: Path, dst: Path, *args, **kwargs):
    """wrapper to shutil.copy2 to support Pathlib."""
    return copy2(str(src), str(dst), *args, **kwargs)


def copytree2(src: Path, dst: Path, *args, **kwargs):
    """wrapper to copytree and cp to support Pathlib, pattern, and override."""
    with tempfile.TemporaryDirectory() as td:
        # hack to support override
        tmpdst = Path(td) / "dst"
        copytree(str(src), str(tmpdst), *args, **kwargs)
        call([
            "/bin/cp",
            # archieve, recursive, force, do not create one inside
            # https://stackoverflow.com/a/24486142/9567349
            "-arfT",
            str(tmpdst),
            str(dst),
        ])


def include_patterns(*include_patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Remove directory starts with _.
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in include_patterns
                   for name in filter(names, pattern))
        removed_dir = any([x.startswith("_") for x in path.split(os.path.sep)])
        ignore = set(name for name in names
                     if (name not in keep or removed_dir) and not os.path.isdir(os.path.join(path, name)))
        return ignore
    return _ignore_patterns


def call(commands: List[str], env={}, **kwargs):
    """Call commands and print to screen for debug.

    Raises
    ------
    RuntimeError
        returned code is not zero
    """
    with sp.Popen(commands, stdout=sys.stdout, stderr=sys.stderr, env=env, **kwargs) as p:
        p.communicate()
        exit_code = p.wait()

        if exit_code:
            raise RuntimeError("Run %s failed, return code: %d" %
                               (" ".join(commands), exit_code))


# the detailed step to build DeePMD-kit

# online resources to download
RESOURCES = {
    # bazelisk is used to warpper bazel
    "bazelisk-1.11.0": OnlineResource(
        "bazel-linux-amd64-1.11.0",
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64",
        "231ec5ca8115e94c75a1f4fbada1a062b48822ca04f21f26e4cb1cd8973cd458",
        executable=True,
    ),
    # tensorflow
    "tensorflow-2.9.1": OnlineResource(
        "tensorflow-2.9.1.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.tar.gz",
        "6eaf86ead73e23988fe192da1db68f4d3828bcdd0f3a9dc195935e339c95dbdc",
        gzip="tensorflow",
    ),
}


class BuildBazelisk(Build):
    def __init__(self, version="1.11.0") -> None:
        self.version = version

    @property
    @lru_cache()
    def resources(self) -> Dict[str, OnlineResource]:
        return {
            "bazelisk": RESOURCES["bazelisk-" + self.version],
        }

    @property
    @lru_cache()
    def dependencies(self) -> Dict[str, Build]:
        return {}

    def build(self):
        bazel_res = self.resources['bazelisk']
        bin_dst = self.prefix / "bin"
        bin_dst.mkdir(exist_ok=True)
        copy3(bazel_res.path, bin_dst / "bazelisk")

    @property
    def built(self):
        return (PREFIX / "bin" / "bazelisk").exists()


class BuildNumpy(Build):
    """Build NumPy"""
    @property
    @lru_cache()
    def resources(self) -> Dict[str, OnlineResource]:
        return {}

    @property
    @lru_cache()
    def dependencies(self) -> Dict[str, Build]:
        return {}

    @property
    def built(self) -> bool:
        return importlib.util.find_spec("numpy") is not None

    def build(self):
        try:
            call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy",
            ])
        except RuntimeError as e:
            raise RuntimeError("Please manually install numpy!") from e


class BuildCUDA(Build):
    """Find CUDA."""
    @property
    @lru_cache()
    def resources(self) -> Dict[str, OnlineResource]:
        return {}

    @property
    @lru_cache()
    def dependencies(self) -> Dict[str, Build]:
        return {}

    def build(self):
        raise RuntimeError(
            "NVCC is not found. Please manually install CUDA"
            "Toolkit and cuDNN!\n"
            "CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive\n"
            "cuDNN: https://developer.nvidia.com/rdp/cudnn-archive")

    @property
    def built(self):
        return CUDA_PATH is not None and CUDNN_PATH is not None

    @property
    def cuda_version(self):
        nvcc_bin = CUDA_PATH / 'bin' / 'nvcc'
        output = sp.check_output([str(nvcc_bin), '--version'], env={}, encoding='utf8').split('\n')
        pattern = re.compile('V[0-9]*\\.[0-9]*\\.[0-9]*')
        for x in output:
            search = pattern.search(x)
            if search is not None:
                # strip "V"
                version = search.group()[1:]
                # only return major and minor
                return ".".join(version.split(".")[:2])
        raise RuntimeError("Not found version in nvcc --version")

    @property
    def cudnn_version(self):
        cudnn_header = CUDNN_PATH / 'include' / 'cudnn.h'
        with open(cudnn_header) as f:
            for line in f:
                if line.startswith("#define CUDNN_MAJOR "):
                    return line.split()[-1]
        cudnn_header = CUDNN_PATH / 'include' / 'cudnn_version.h'
        with open(cudnn_header) as f:
            for line in f:
                if line.startswith("#define CUDNN_MAJOR "):
                    return line.split()[-1]
        raise RuntimeError(
            "cuDNN version is not found!\n"
            "Download from: https://developer.nvidia.com/rdp/cudnn-archive"
            )

    @property
    @lru_cache()
    def cuda_compute_capabilities(self):
        """Get cuda compute capabilities."""
        cuda_version = tuple(map(int, self.cuda_version.split(".")))
        if (10, 0, 0) <= cuda_version < (11, 0, 0):
            return "sm_35,sm_50,sm_60,sm_62,sm_70,sm_72,sm_75,compute_75"
        elif (11, 0, 0) <= cuda_version < (11, 1, 0):
            return "sm_35,sm_50,sm_60,sm_62,sm_70,sm_72,sm_75,sm_80,compute_80"
        elif (11, 1, 0) <= cuda_version:
            return "sm_35,sm_50,sm_60,sm_62,sm_70,sm_72,sm_75,sm_80,sm_86,compute_86"
        else:
            raise RuntimeError("Unsupported CUDA version")


class BuildTensorFlow(Build):
    """Build TensorFlow C++ interface.

    Parameters
    ----------
    version : str, default=2.9.1
        TensorFlow version
    enable_mkl : bool, default=True
        enable OneDNN
    enable_cuda : bool, default=False
        Enable CUDA build
    """

    def __init__(self, version: str ="2.9.1", enable_mkl: bool=True, enable_cuda: bool=False) -> None:
        self.version = version
        self.enable_mkl = enable_mkl
        self.enable_cuda = enable_cuda

    @property
    @lru_cache()
    def resources(self) -> Dict[str, OnlineResource]:
        return {
            "tensorflow": RESOURCES["tensorflow-" + self.version],
        }

    @property
    @lru_cache()
    def dependencies(self) -> Dict[str, Build]:
        optional_dep = {}
        if self.enable_cuda:
            optional_dep['cuda'] = BuildCUDA()
        return {
            "bazelisk": BuildBazelisk(),
            "numpy": BuildNumpy(),
            **optional_dep,
        }

    def build(self):
        tf_res = self.resources['tensorflow']
        src = tf_res.gzip_path / ("tensorflow-%s" % self.version)
        with set_directory(src):
            # configure -- need bazelisk in PATH
            call([str(src / "configure")], env={
                "PATH": list2env([PREFIX / "bin", "/usr/bin"]),
                **self._environments,
            })
            # bazel build
            call([
                str(PREFIX / "bin" / "bazelisk"),
                *self._bazel_opts,
                "build",
                *self._build_opts,
                *self._build_targets,
            ], env={
                "PATH": list2env(["/usr/bin"]),
                "HOME": os.environ.get("HOME"),
                "TEST_TMPDIR": str(PACKAGE_DIR / "bazelcache"),
                # for libstdc++
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                "CC": str(Path(GCC).resolve()),
                "CXX": str(Path(GXX).resolve()),
            })

        # copy libraries and directories
        ext = get_shlib_ext()
        lib_dst = self.prefix / "lib"
        include_dst = self.prefix / "include"
        lib_dst.mkdir(exist_ok=True)
        include_dst.mkdir(exist_ok=True)

        # 1. copy headers
        (include_dst / "tensorflow").mkdir(exist_ok=True)
        copytree2(src / "tensorflow" / "cc", include_dst /
                  "tensorflow" / "cc", ignore=include_patterns('*.h', '*.inc'))
        copytree2(src / "tensorflow" / "core", include_dst /
                  "tensorflow" / "core", ignore=include_patterns('*.h', '*.inc'))
        # bazel-bin includes generated headers like version, pb.h, ..
        copytree2(src / "bazel-bin", include_dst,
                  ignore=include_patterns('*.h', '*.inc'))

        copytree2(src / "third_party", include_dst /
                  "third_party", ignore=ignore_patterns('*.cc'))
        bazel_tensorflow = src / ("bazel-" + src.name)
        copytree2(bazel_tensorflow / "external" /
                  "eigen_archive" / "Eigen", include_dst / "Eigen")
        copytree2(bazel_tensorflow / "external" / "eigen_archive" /
                  "unsupported", include_dst / "unsupported")
        copytree2(bazel_tensorflow / "external" / "com_google_protobuf" /
                  "src" / "google", include_dst / "google")
        copytree2(bazel_tensorflow / "external" /
                  "com_google_absl" / "absl", include_dst / "absl")

        # 2. copy libraries
        if self.enable_mkl:
            copy3(src / "bazel-out" / "k8-opt" / "bin" / "external" /
                "llvm_openmp" / ("libiomp5" + ext), lib_dst)
        lib_src = src / "bazel-bin" / "tensorflow"
        self.copy_lib("libtensorflow_framework" + ext, lib_src, lib_dst)
        self.copy_lib("libtensorflow_cc" + ext, lib_src, lib_dst)

    def copy_lib(self, libname, src, dst):
        """Copy library and make symlink."""
        copy3(src / (libname + "." + self.version), dst)
        libname_v = libname + "." + self.version
        (dst / (libname + "." + self.version.split(".")
         [0])).symlink_to(libname_v)
        (dst / libname).symlink_to(libname_v)

    @property
    def _environments(self) -> dict:
        if self.enable_cuda:
            cuda_env = {
                "TF_NEED_CUDA": "1",
                # /usr is path to driver
                "TF_CUDA_PATHS": ",".join((str(CUDA_PATH), str(CUDNN_PATH), "/usr")),
                "TF_CUDA_VERSION": str(self.dependencies['cuda'].cuda_version),
                "TF_CUDNN_VERSION": str(self.dependencies['cuda'].cudnn_version),
                "TF_NCCL_VERSION": "",
                "TF_CUDA_COMPUTE_CAPABILITIES": self.dependencies['cuda'].cuda_compute_capabilities,
                "GCC_HOST_COMPILER_PATH": str(Path(GCC).resolve()),
                "GCC_HOST_COMPILER_PREFIX": str(Path(GCC).resolve().parent.parent),
            }
        else:
            cuda_env = {
                "TF_NEED_CUDA": "0",
            }
        return {
            "TF_ENABLE_XLA": "1",
            "CC_OPT_FLAGS": "-Wno-sign-compare",
            # Python settings
            "PYTHON_BIN_PATH": sys.executable,
            "USE_DEFAULT_PYTHON_LIB_PATH": "1",
            # Additional settings
            "TF_NEED_OPENCL": "0",
            "TF_NEED_OPENCL_SYCL": "0",
            "TF_NEED_COMPUTECPP": "0",
            "TF_CUDA_CLANG": "0",
            "TF_NEED_TENSORRT": "0",
            "TF_NEED_ROCM": "0",
            "TF_NEED_MPI": "0",
            "TF_DOWNLOAD_CLANG": "0",
            "TF_SET_ANDROID_WORKSPACE": "0",
            "TF_CONFIGURE_IOS": "0",
            ** cuda_env,
        }

    @property
    def _build_targets(self) -> List[str]:
        # C++ interface
        return ["//tensorflow:libtensorflow_cc" + get_shlib_ext()]

    @property
    def _build_opts(self) -> List[str]:
        opts = [
            "--logging=6",
            "--verbose_failures",
            "--config=opt",
            "--config=noaws",
            "--copt=-mtune=generic",
            "--local_cpu_resources=%d" % CPU_COUNT,
        ]
        if self.enable_mkl:
            # enable oneDNN
            opts.append("--config=mkl")
        return opts

    @property
    def _bazel_opts(self) -> List[str]:
        return []

    @property
    def built(self):
        return (PREFIX / "lib" / ("libtensorflow_cc%s.%s" % (get_shlib_ext(), self.version))).exists()


def clean_package():
    """Clean the unused files."""
    clean_files = [
        PACKAGE_DIR,
        # bazelisk
        PREFIX / "bin" / "bazelisk",
        # numpy
        PREFIX / "numpy",
        # bazel cache
        Path.home() / ".cache" / "bazel",
    ]
    for f in clean_files:
        shutil.rmtree(str(f), ignore_errors=True)


# interface

def env() -> Dict[str, str]:
    return {
        "Python": sys.executable,
        "CUDA": CUDA_PATH,
        "cuDNN": CUDNN_PATH,
        "gcc": GCC,
        "g++": GXX,
        "Install prefix": PREFIX,
        "Packages": PACKAGE_DIR,
    }


def pretty_print_env() -> str:
    return ("Build configs:\n" +
            "\n".join(["%s:%s%s" % (kk, " "*(19-len(kk)), vv) for kk, vv in env().items() if vv is not None]))


class RawTextArgumentDefaultsHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse_args(args: Optional[List[str]] = None):
    """TensorFlow C++ Library Installer commandline options argument parser.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Installer of Tensorflow C++ Library.\n\n" + pretty_print_env(),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix of installed paths.",
    )
    parser.add_argument(
        "--packages",
        type=str,
        default=str(PACKAGE_DIR),
        help="Path to download packages.",
    )
    parser.add_argument(
        "--cuda",
        action='store_true',
        help="Enable CUDA for TensorFlow and DeePMD-kit",
    )
    parser.add_argument(
        "--cuda-path",
        type=str,
        default=CUDA_PATH,
        help="path to CUDA Toolkit",
    )
    parser.add_argument(
        "--cudnn-path",
        type=str,
        default=CUDNN_PATH,
        help="path to cuDNN",
    )
    parser.add_argument(
        "--gcc",
        type=str,
        default=GCC,
        help="path to gcc",
    )
    parser.add_argument(
        "--gxx",
        type=str,
        default=GXX,
        help="path to gxx",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=CPU_COUNT,
        help="Number of CPU cores used to build.",
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean files after build.',
    )
    parsed_args = parser.parse_args(args=args)

    return parsed_args


def str_to_path_if_not_none(x: str) -> Path:
    if x is not None:
        return Path(x).absolute()
    return x


if __name__ == "__main__":
    args = parse_args()
    # override default settings
    PREFIX = str_to_path_if_not_none(args.prefix)
    PACKAGE_DIR = str_to_path_if_not_none(args.packages)
    CPU_COUNT = args.cpus
    CUDA_PATH = str_to_path_if_not_none(args.cuda_path)
    CUDNN_PATH = str_to_path_if_not_none(args.cudnn_path)
    GCC = args.gcc
    GXX = args.gxx
    assert GCC is not None
    assert GXX is not None

    dlog.info(pretty_print_env())

    # create directories
    PACKAGE_DIR.mkdir(exist_ok=True)
    PREFIX.mkdir(exist_ok=True)

    # start to build
    BuildTensorFlow(enable_cuda=args.cuda)()
    dlog.info("Build TensorFlow C++ Library successfully!")

    # clean
    if args.clean:
        clean_package()

