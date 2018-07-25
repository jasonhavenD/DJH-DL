- 只有方阵才有矩阵的逆，不可逆的矩阵-奇异矩阵

- 梯度下降速度更快，首先要确定梯度下降正确运行（J缓慢下降）

  - 特征缩放
    - 常用方法：均值归一化、方差归一化
  - 学习率
    - 固定
      - 如果太小，缓慢收敛
      - 如果太大，也许不会正常运行
      - 需要测试选择
    - 动态变化，收敛测试

- 正规方程normal equation

  - $$
    \theta = (X^TX)^{-1}X^Ty
    $$

    - 如果（X^TX）不可逆
      - 很少发生
      - 可能特征数目太多导致，可以删除一些特征，或者使用正则化

- 如果数据特征很多，使用gradient descent，否则推荐使用正规方程

- logistic function == sigmoid function

  - linear regression ： h(X)=W^TX
  - sigmoid fucntion : g=g(z)=1/(1+exp(-z))
  - logistic function : h(X)=g(W^TX)=1/(1+exp(-W^TX))

- 决策边界 decision boundary

- 代价函数 cost function

  - Linear Regression

    > Cost=1/2*(h(x)-y)^2 损失函数
    >
    > J=sum(Cost) 代价函数，sum(损失函数)

  - Logistic Regression

    - 直接将logistics function 代入损失函数和J函数，J函数是非凸函数，所以需要重新设计cost function

    - cost function

      - 分段函数 凸函数

      - $$
        cost\left( h_{\theta }\left( x\right) ,y\right) =\begin{cases}-\log \left( h_{\theta }\left( x\right) \right) y=1\\
        -\log \left( 1- h_{\theta }\left( x\right) \right)  y=0\end{cases}
        $$

      - 简化cost function
        $$
        cost(h_{\theta}(x),y)=-y*\log \left( h_{\theta }\left( x\right) \right)-(1-y)*\log \left( 1- h_{\theta }\left( x\right) \right)
        $$

    - J function 凸函数

      - $$
        J(\theta)=\sum_{i=1}^{m} cost(h_\theta(x)^{i},y^{i})
        $$

  - 高级优化，计算J和偏导数

    - Gradient Desent
    - Conjugate gradient
    - BFGS
    - LBFGS
      - 速度快，自动调整学习率
      - 比梯度下降更加复杂

  - 过拟合问题

    - r减少特征数目
    - 正则化
      - 代价函数+正则项 
      - 正则项用来缩小全部参数取值，降低模型复杂度
      - 正则项的参数用来控制不同部分的取舍，达到平衡，不能取值太大，会导致欠拟合

  - 神经网络

    - 源于人脑模拟
    - 由于计算量很大，限于计算机的发展而沉睡很多年
    - 如何表示神经网络模型
    - 神经元 == 一个逻辑单元
    - neuron model:logistic unit
    - 有时候会需要一个偏置神经元 
    - 激活函数 == 非线性函数
    - 模型参数 == 模型权重
    - 神经网络就是一组神经元
    - 前向传播过程
    - 多分类问题
      - one vs all 每个类别用一个逻辑回归神经元
    - 神经网络的代价函数，参数拟合
    - 反向传播，代价函数最小化
      - ​



















