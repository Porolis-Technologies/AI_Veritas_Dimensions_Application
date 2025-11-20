from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "requirements.txt", "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

PACKAGE_NAME = "deploy"

exec(open(BASE_DIR / PACKAGE_NAME / "version.py").read())

# Define the package
setup(
    name=PACKAGE_NAME,
    version=__version__,
    python_requires=">=3.10",
    packages=find_namespace_packages(),
    install_requires=required_packages,
)