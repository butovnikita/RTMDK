"""Setup script for RTMDK package."""

from setuptools import setup, find_packages

setup(
    name="rtmdk",
    version="8.0.0",
    description="Resonance-Topological Memory for LLMs",
    author="RTMDK Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "sentence-transformers>=2.2.0",
        "msgpack>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rtmdk=rtmdk.main:main",
            "rtmdk-proxy=rtmdk.st_proxy:main",
        ],
    },
)
