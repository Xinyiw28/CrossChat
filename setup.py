"""Setup."""
from setuptools import find_packages
from setuptools import setup

install_requires = [
    "numpy",
    "networkx",
    "matplotlib",
    "pandas",
    "scipy",
    "scanpy",
]

setup(
    name='CrossChat',
    version='0.0.1',
    install_requires=install_requires,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url=None,
    license='MIT',
    author='Xinyi',
    author_email='xinyiw28@uci.edu',
    description='This package aims to detect and analyze hierarchical structures within cell-cell communications (CCC).'
)
