# setup.py
from setuptools import setup, find_packages

setup(
    name="AutoDroneChallenge",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy", "optuna", "matplotlib", "pillow"],
)
