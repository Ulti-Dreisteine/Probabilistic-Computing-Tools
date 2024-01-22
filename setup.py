from setuptools import setup, find_packages
from pathlib import Path

VERSION = "1.0.0"
DESCRIPTION = "Package for information estimation, independence test, causal structure mining, etc."
this_directory = Path(__file__).parent
long_description = (this_directory/"project_description.md").read_text(encoding="utf-8")

setup(
    name="giefstat",
    version=VERSION,
    author="Dreisteine",
    author_email="dreisteine@163.com",
    description=DESCRIPTION,
    packages=find_packages(include=["giefstat", "giefstat.*"]),  # NOTE: 注意giefstat.*的写法
    install_requires=[],
    keywords=["python", "information estimation", "time series", "transfer entropy"],
    python_requires=">=3.8.0",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ulti-Dreisteine/general-information-estimation-framework",
)