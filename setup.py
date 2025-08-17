"""Setup script for EFI package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="embodied-field-intelligence",
    version="0.1.0",
    author="EFI Team",
    description="CA-based framework for embodied artificial intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/embodied-field-intelligence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "gym": ["gymnasium>=0.26.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "ffmpeg-python>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "efi=cli:main",
        ],
    },
)