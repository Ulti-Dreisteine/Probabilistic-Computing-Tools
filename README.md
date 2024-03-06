### 概率计算工具

#### 环境配置

```
conda create -c conda-forge -n statistic_computation python=3.8 "pymc>=4"
```

#### 依赖包

```bash
pip list --format=freeze > requirements.txt
```

#### 包的发布

```bash
python setup.py sdist bdist_wheel
python setup.py install

pip install twine
twine upload dist/*
```

