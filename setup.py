from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="docintel",
    version="0.1.0",
    author="Alseitov Olzhas",
    author_email="olzhas010111@gmail.com",
    description="An AI-powered document processing and analytics platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InfiniteJas/docintel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.68.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "spacy>=3.1.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pytesseract>=0.3.8",
        "PyMuPDF>=1.18.0",
        "yake>=0.4.0",
        "scikit-learn>=0.24.0",
        "sentence-transformers>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15.0",
            "pytest-mock>=3.6.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
    },
)
