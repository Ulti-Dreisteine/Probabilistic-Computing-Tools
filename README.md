### 概率计算工具

#### 环境配置

```
conda create -c conda-forge -n stat_comput python=3.8 "pymc>=4"
```

dependencies: requiments.txt

#### Package Info

https://pypi.org/search/?q=giefstat

#### Project Purpose

This project aims to lay a basis for:
1. computing higher-order information interactions between different types (discrete & continuous) of variables
2. uncovering complex associations and causal relationships in high-dimensional, nonlinear and nonstationary data

#### Environment Setup

```bash
# 创建环境
conda create -n giefstat python=3.9

# 激活环境
conda activate giefstat

# 安装依赖
pip install jupyter matplotlib==3.7.3 pandas==1.5.3 numpy==1.24.4 pingouin==0.5.4 scikit-learn==0.24.0 scipy==1.10.1 arviz==0.15.1 category_encoders==2.6.3

# 安装本地wheel文件
pip install F:\github\Probabilistic-Computing-Tools\pkg\minepy-1.2.6-cp39-cp39-win_amd64.whl
```

#### Project Structure

```
    |-- giefstat
    |   |
    |   |-- __init__.py
    |   |-- setting.py                      # 项目设置
    |   |-- util.py                         # 通用工具
    |   |
    |   |-- coefficient                     # 基于KNN和KDE等方法的数据信息计算和关联估计方法
    |   |   |-- __init__.py
    |   |   |
    |   |   |-- corr_coeff                  # 常见相关系数
    |   |   |   |-- __init__.py
    |   |   |   |-- coeff.py                # Pearson系数、Spearman系数和距离相关系数
    |   |   |
    |   |   |-- mi_gief                     # 通用信息估计
    |   |   |   |-- __init__.py
    |   |   |   |-- entropy                 # 信息熵
    |   |   |   |   |-- __init__.py
    |   |   |   |   |-- cond_entropy.py     # 条件熵估计
    |   |   |   |   |-- marg_entropy.py     # 边际熵估计
    |   |   |   |-- mutual_info
    |   |   |       |-- __init__.py
    |   |   |       |-- _kraskov.py         # 由Kraskov等提出的K近邻互信息估计
    |   |   |       |-- _ross.py            # 由Ross等提出的互信息估计
    |   |   |       |-- mi.py               # 互信息估计
    |   |   |       |-- cmi.py              # 条件互信息估计
    |   |   |
    |   |   |-- mi_kde                      # 基于KDE的边际熵和互信息估计
    |   |   |   |-- __init__.py
    |   |   |   |-- kde.py  
    |   |   |
    |   |   |-- mic                         # 最大信息系数
    |   |   |   |-- __init__.py
    |   |   |   |-- _mic_rmic.py            # MIC和RMIC计算
    |   |   |   |-- mi_cmi.py               # 基于MIC和RMIC的互信息和条件互信息估计
    |   |   |   |-- rgsr.pickle             # RMIC中用于修正MIC下界的回归模型
    |   |   |
    |   |   |-- mi_model
    |   |   |   |-- __init__.py
    |   |   |   |-- mi_cmi.py               # 基于机器学习预测模型的关联和条件关联系数估计
    |   |   |
    |   |   |-- mi_quant
    |   |       |-- __init__.py
    |   |       |-- _quant_darbellay.py     # Darbellay数据离散化方法
    |   |       |-- mi_classic.py           # 基于经典等距和等频离散化的互信息估计
    |   |       |-- mi_darbellay.py         # 基于Darbellay离散化的互信息估计
    |   |   
    |   |-- indep_test
    |   |   |-- __init__.py
    |   |   |-- surrog_indep_test.py        # 基于Bootstrap的关联度量和独立性检验
    |   |
    |   |-- time_series                     # 时序关联和因果挖掘
    |   |   |-- __init__.py
    |   |   |-- util.py                     # 序列符号化、时延峰解析等工具
    |   |   |-- td_assoc_analysis.py        # 成对时延关联分析
    |   |   |-- transfer_entropy.py         # 成对时延传递熵检验
    |   |   |-- partial_transfer_entropy.py # 成对时延偏传递熵检验
    |   
    |-- test                                # 对应方法的单元测试和应用案例
    |   |-- coefficient
    |   |   |-- corr_coeff
    |   |   |   |-- test.py                 # unittest
    |   |   |-- mi_gief
    |   |   |   |-- test.py                 # unittest
    |   |   |-- mi_kde
    |   |   |   |-- test.py                 # unittest
    |   |   |-- mi_model
    |   |   |-- |-- test.py                 # unittest
    |   |   |-- mi_quant
    |   |   |-- |-- test.py                 # unittest
    |   |   |-- mic
    |   |   |-- |-- test.py                 # unittest
    |   |
    |   |-- independence_test
    |   |   |-- test_surrog_indep_test.py   # 案例测试
    |   |
    |   |-- time_series
    |   |   |-- test_real_td_assoc_analysis.py  # 案例测试
    |   |   |-- test_simple_td_assoc_analysis.py    # 案例测试
    |   |   |-- test_transfer_entropy_cyclic.py # 案例测试
    |   |   |-- test_transfer_entropy_siso.py   # 案例测试
```
   
#### Notes

1. <font color="red">根据FGD测试结果, 离散变量可被stdize_values处理后视为连续变量, 代入MI-GIEF中进行计算</font>
2. <font color="red">stdize_values在对连续变量处理过程时加入了噪音并归一化</font>


#### References

1. A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information. Physical Review E, 2003.
2. D. Lombardi, S. Pant: A Non-Parametric K-Nearest Neighbor Entropy Estimator. Physical Review E, 2015.
3. B. C. Ross: Mutual Information between Discrete and Continuous Data Sets. PLoS One, 2014.
4. https://github.com/dizcza/entropy-estimators
5. https://github.com/danielhomola/mifs

#### Todos

1. 紧凑时序因果挖掘
2. 基于贝叶斯网络的独立性检验

#### 包的发布

```bash
python setup.py sdist bdist_wheel
python setup.py install

pip install twine
twine upload dist/*
```

