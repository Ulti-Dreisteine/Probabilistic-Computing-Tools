from setuptools import setup, find_packages
from pathlib import Path

VERSION = "0.0.1"
DESCRIPTION = "Package for information estimation, independence test, causal structure mining, etc."
this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text(encoding="utf-8")

setup(
    name="gen_info_est_tools",
    version=VERSION,
    author="Dreisteine",
    author_email="dreisteine@163.com",
    description=DESCRIPTION,
    packages=find_packages(include=["giefstat", "giefstat.*"]),  # NOTE: 注意giefstat.*的写法
    install_requires=["numpy", "pandas", "scipy", "statsmodels", "tqdm", "matplotlib", "seaborn", "networkx", "pygraphviz", "pydot", "graphviz", "pyinform", "pyitlib", "pyts", "sklearn", "pyts"],
    keywords=["python", "information estimation", "time series", "transfer entropy"],
    python_requires=">= 3.8",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ulti-Dreisteine/Probabilistic-Computing-Tools",
)