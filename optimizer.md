# The most common optimization algorithms

- [paper](https://arxiv.org/pdf/1609.04747.pdf)

## 1.Gradient Descent

- Gradient descent is a way to minimize an objective function J(θ) parameterized by a model’s parameters θ ∈ R  by updating the parameters in the opposite direction of the gradient of the objective function ∇θJ(θ).
- The learning rate η determines the size of the steps we take to reach a (local) minimum
- In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley.
- there are **three variants **of gradient descent, which differ in **how much data we use to compute the gradient** of the obejctive function. Depending on the amount of data, we make a trade-off between the **accuracy** of the parameter update and the **time** it takes to perform an update.

##### 1.1BGD

- Batch gradient descent
- θ = θ − η · ∇θJ(θ) 

- computes the gradient of the cost function  to the parameters θ for **the entire training dataset**

- Batch gradient descent is guaranteed to **converge to the global minimum for convex error surfaces **and to **a local minimum for non-convex surfaces**.

-  **very slow** , is intractable for datasets that **do not fit in memory**,not allow us to **update our model online**,e, i.e. with new examples on-the-fly.

- ```python
  # In code, batch gradient descent looks something like this:
  for i in range ( nb_epochs ):
  	params_grad = evaluate_gradient ( loss_function , data , params )
  	params = params - learning_rate * params_grad
  ```

##### 1.2SGD

- Stochastic gradient descent

- θ = θ − η · ∇θJ(θ; x(i); y(i)) 

- Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example
  **(x(i),y(i))**

- enables it to jump to **new and potentially better local minima**

- **faster**,**can  be used to learn online.**

- ```python
  #Its code fragment simply adds a loop over the training examples and evaluates the gradient w.r.t. each example.
  for i in range ( nb_epochs ):
  	np . random . shuffle ( data )#Note that we shuffle the training data at every epoch
  	for example in data :
  		params_grad = evaluate_gradient ( loss_function , example , params )
  		params = params - learning_rate * params_grad
  ```

##### 1.3 Mini-batch gradient descent

- Mini-batch gradient descent finally takes the best of both worlds and performs an update for every
  mini-batch of n training examples

- θ = θ − η · ∇θJ(θ; x(i:i+n); y(i:i+n)) 

- a) reduces the variance of the parameter updates, which can lead to more stable convergence

- b) can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient

- ```python
  #In code, instead of iterating over examples, we now iterate over mini-batches of size 50:
  for i in range ( nb_epochs ):
  	np . random . shuffle ( data )
  	for batch in get_batches ( data , batch_size =50):
  		params_grad = evaluate_gradient ( loss_function , batch , params )
  		params = params - learning_rate * params_grad
  ```

-  does not guarantee good convergence

#### Challenges

1. Choosing a proper learning rate can be difficult. 
2. Learning rate schedules  try to adjust the learning rate during training, reducing the learning rate according to a pre-defined schedule or when the change in objective between epochs falls below a threshold
3. the same learning rate applies to all parameter updates
4. Another key challenge of minimizing highly non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima.

## 2.Momentum(动量)

- SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one
  dimension than in another , which are common around local optima

- Momentum is a method that **helps accelerate SGD** in the relevant direction and dampens （抑制）oscillations, It does this by adding a fraction γ of the update vector of the past time step to the current update vector.

- > vt = γvt−1 + η∇θJ(θ) The momentum term γ is usually set to 0.9 or a similar value.
  >
  > θ = θ − vt

- The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.

- 动量方法主要是为了解决Hessian矩阵病态条件问题（直观上讲就是梯度高度敏感于参数空间的某些方向）。

- ​

## 3. NAG(牛顿动量)

- Nesterov Accelerated Gradient

- NAG is a way to give our momentum term this kind of prescience.

- > vt = γ vt−1 + η∇θJ(θ − γvt−1)
  > θ = θ − vt

- Nesterov是[Momentum](http://blog.csdn.net/bvl10101111/article/details/72615621)的变种。

- 与Momentum唯一区别就是，计算梯度的不同，Nesterov先用当前的速度v更新一遍参数，在用更新的临时参数计算梯度。

- 相当于添加了矫正因子的Momentum。

## 4.Adagrad

- It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. 

- it is well-suited for dealing with sparse data.

- $$
  \theta_{t+1}=\theta_t- \frac{\eta}{\sqrt{G_t+\epsilon}}\nabla_{\theta_t} J(\theta)
  $$

- Adagrad uses a different learning rate for every parameter θi at every time step t

- In its update rule, Adagrad modifies the general learning rate η at each time step t for every parameter
  θi based on the past gradients that have been computed for θi

- **main benefit ** is it eliminates the need to manually tune the learning rate.Most implementations use a default value of 0.01 and leave it at that.

- **main weakness** is its accumulation of the squared gradients in the denominator,Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge.

## 5.Adadelta

- Adadelta  is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing
  learning rate. Instead of accumulating all past squared gradients, Adadelta **restricts the window of accumulated past gradients to some fixed size w.**

- With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the
  update rule.

- > ∆θt = (−(RMS[∆θ]t−1)/(RMS[g]t))*gt
  >
  > θt+1 = θt + ∆θt

## RMSprop

- RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton  in Lecture 6e
  of his Coursera Class
- RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad’s radically diminishing learning rates. RMSprop in fact is identical to the first update vector of Adadelta
- RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.
- Hinton suggests γ to be set to 0.9, while a good default value for the learning rate η is 0.001

## Adam

- Adaptive Moment Estimation

- is another method that computes adaptive learning rates for each parameter. 

- suggest : alpha=0.001、beta1=0.9、beta2=0.999 和 epsilon=10E−8

- The authors propose default values of 0.9 for β1, 0.999 for β2, and 10−8 for r . They show empirically

  that Adam works well in practice and compares favorably to other adaptive learning-method algorithms.

## How to Choose

So, which optimizer should you use? 

If your **input data is sparse**, then you likely achieve the best results using one of **the adaptive learning-rate methods**. An additional benefit is that you will not need to tune the learning rate but will likely achieve the best results with the default value.
In summary, **RMSprop** is an extension of **Adagrad** that deals with **its radically diminishing learning rates** . It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. **Adam**, finally, **adds bias-correction and momentum to RMSprop**. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances.
Kingma et al. [10] show that its bias-correction helps **Adam** slightly outperform RMSprop towards
the end of optimization as gradients become sparser. **Insofar, Adam might be the best overall choice.**

Interestingly, many recent papers use vanilla SGD without momentum and a simple learning rate annealing schedule. As has been shown, **SGD usually achieves to find a minimum**, but it might **take significantly longer ** than with some of the optimizers, is **much more reliant on a robust initialization and annealing schedule** , and may get stuck in saddle points rather than local minima. 

Consequently, if you care about **fast convergence and train a deep or complex neural network**, you should choose one of the **adaptive learning rate methods**.