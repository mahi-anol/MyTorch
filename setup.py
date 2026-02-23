from setuptools import setup, find_packages

setup(
    name="MyTorch",
    version="0.1.0",
    description="A light-weight deep learning library for simulating pytorch like functionality. ",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pytest>=6.0",
        "hypothesis>=6.0",
    ],
    extras_require={
        "viz": ["streamlit>=1.12.0", "matplotlib>=3.5.0", "numpy>=1.21.0"],
        "dev": ["black>=22.0", "flake8>=4.0"],
    },
)