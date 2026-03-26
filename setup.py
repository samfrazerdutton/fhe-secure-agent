from setuptools import setup, find_packages

setup(
    name="fhe-secure-agent",
    version="0.1.0",
    author="Sam Frazer-Dutton",
    description="GPU-accelerated FHE security layer for any LLM agent",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "cupy-cuda12x",
    ],
)
