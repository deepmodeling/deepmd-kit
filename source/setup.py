from skbuild import setup
from os import path
import imp

readme_file = path.join(path.dirname(path.abspath(__file__)), '..', 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    with open(readme_file) as f:
        readme = f.read()


tf_install_dir = imp.find_module('tensorflow')[1] 

# install_requires = ['xml']
install_requires=[]

setup(
    name="deepmd-kit",
    setup_requires=['setuptools-git-version'],
    version_format='{tag}.dev{commitcount}+{gitsha}',
    author="Han Wang",
    author_email="wang_han@iapcm.ac.cn",
    description="A deep learning package for many-body potential energy representation and molecular dynamics",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmodeling/deepmd-kit",
    packages=['deepmd'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords='deepmd',
    install_requires=install_requires,        
    cmake_args=['-DTENSORFLOW_ROOT:STRING=%s' % tf_install_dir, 
                '-DTF_GOOGLE_BIN:BOOL=FALSE', 
                '-DBUILD_PY_IF:BOOL=TRUE', 
                '-DBUILD_CPP_IF:BOOL=FALSE',
    ],
)
