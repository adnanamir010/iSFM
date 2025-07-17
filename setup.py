from setuptools import setup, find_packages

setup(
    name="hybrid_sfm",
    version="0.1.0",
    description="Hybrid Structure-from-Motion with Points, Lines, and Vanishing Points",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
    ],
)