"""
ROX Proven Edge Engine v3.0
Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rox-proven-edge-engine",
    version="3.0.0",
    author="ROX Trading Systems",
    author_email="support@roxtrading.com",
    description="Multi-Agent AI-Powered Swing Trading System for Indian Stock Market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roxtrading/proven-edge-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "colorlog>=6.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "data": [
            "requests>=2.31.0",
            "ta-lib>=0.4.28",
        ],
    },
    entry_points={
        "console_scripts": [
            "rox-engine=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
