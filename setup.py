import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="submission",
    version="0.0.1",
    packages=find_packages(),
    description="",
    long_description=read("README.md"),
    install_requires=[
        "ray[rllib]>=1.2.0",
        "torch>=1.6.0,<=1.7.0",
        "tensorboard==1.15.0",
        "numba>=0.51.2",
        "matplotlib>=3.3.2",
        "ordered-set",
        "click",
        "gym>=0.10.5",
        "numpy>=1.11",
        "dm-sonnet==1.20",
        "tensorflow>=1.8.0,<2.0.0",
        "trueskill",
        "seaborn==0.9.0",
        "psutil",
        "cmake",
        "setuptools",
        "tqdm",
        "pytest",
        "pytest-cov",
        "flake8",
        "ipython",
        "notebook",
        "jupyter_contrib_nbextensions",
        "flaky",
        "pytest-xdist",
    ],
)
