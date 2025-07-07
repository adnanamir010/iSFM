from setuptools import setup, find_packages

setup(
    name="hybrid_sfm",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True, # This tells it to use MANIFEST.in
    python_requires=">=3.9",
)