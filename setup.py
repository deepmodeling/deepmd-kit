from skbuild import setup
from skbuild.exceptions import SKBuildError
from skbuild.cmaker import get_cmake_version
from setuptools_scm import get_version
from packaging.version import LegacyVersion
from os import path, makedirs
import imp, sys, platform

def get_dp_install_path() :
    site_packages_path = path.join(path.dirname(path.__file__), 'site-packages')
    dp_scm_version     = get_version(root="./", relative_to=__file__)
    python_version     = 'py' + str(sys.version_info.major + sys.version_info.minor * 0.1)
    os_info            = sys.platform
    machine_info       = platform.machine()
    dp_pip_install_path    = site_packages_path + '/deepmd'
    dp_setup_install_path    = site_packages_path + '/deepmd_kit-' + dp_scm_version + '-' + python_version + '-' + os_info + '-' + machine_info + '.egg/deepmd'
    
    return dp_pip_install_path, dp_setup_install_path

readme_file = path.join(path.dirname(path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    with open(readme_file) as f:
        readme = f.read()

try:
    tf_install_dir = imp.find_module('tensorflow')[1] 
except ImportError:
    site_packages_path = path.join(path.dirname(path.__file__), 'site-packages')
    tf_install_dir = imp.find_module('tensorflow', [site_packages_path])[1]

install_requires=['numpy', 'scipy']
setup_requires=['setuptools_scm', 'scikit-build', 'cmake']

# add cmake as a build requirement if cmake>3.0 is not installed
try:
    if LegacyVersion(get_cmake_version()) < LegacyVersion("3.0"):
        setup_requires.append('cmake')
except SKBuildError:
    setup_requires.append('cmake')

try:
    makedirs('deepmd')
except OSError:
    pass

dp_pip_install_path, dp_setup_install_path = get_dp_install_path()

setup(
    name="deepmd-kit",
    setup_requires=setup_requires,
    use_scm_version={'write_to': 'deepmd/_version.py'},
    author="Han Wang",
    author_email="wang_han@iapcm.ac.cn",
    description="A deep learning package for many-body potential energy representation and molecular dynamics",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmodeling/deepmd-kit",
    packages=['deepmd'],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords='deepmd',
    install_requires=install_requires,        
    cmake_args=['-DTENSORFLOW_ROOT:STRING=%s' % tf_install_dir, 
                '-DBUILD_PY_IF:BOOL=TRUE', 
                '-DBUILD_CPP_IF:BOOL=FALSE',
                '-DFLOAT_PREC:STRING=high',
                '-DDP_PIP_INSTALL_PATH=%s' % dp_pip_install_path,
                '-DDP_SETUP_INSTALL_PATH=%s' % dp_setup_install_path,
    ],
    cmake_source_dir='source',
    cmake_minimum_required_version='3.0',
    extras_require={
        'test': ['dpdata>=0.1.9'],
    },
    entry_points={
          'console_scripts': ['dp = deepmd.main:main']
    }
)
