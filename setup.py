from setuptools import setup, find_packages
from pathlib import Path

VERSION = "1.1.0"
DESCRIPTION = "Package for information estimation, independence test, causal structure mining, etc."
this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text(encoding="utf-8")

setup(
    name="giefstat",
    version=VERSION,
    author="Dreisteine",
    author_email="dreisteine@163.com",
    description=DESCRIPTION,
    packages=find_packages(include=["giefstat", "giefstat.*"]),  # NOTE: 注意giefstat.*的写法
    install_requires=[
        "arviz==0.15.1",
        "category_encoders==2.6.3",
        "matplotlib==3.7.3",
        "minepy==1.2.6",
        "pandas==1.5.3",
        "numpy==1.24.4",
        "pingouin==0.5.4",
        "pyitlib==0.2.3",
        "scikit_learn==0.24.0",
        "scipy==1.10.1",
        ],
    keywords=["python", "information estimation", "time series", "transfer entropy"],
    python_requires="== 3.8.8",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ulti-Dreisteine/Probabilistic-Computing-Tools",
)