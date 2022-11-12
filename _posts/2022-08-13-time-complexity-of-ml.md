---
layout: post
title: "Time and Space Complexity of Machine Learning Models"
date: 2022-08-13
category: Machine Learning
image: 
excerpt: We detail the time complexity of machine learning models.
katex: True
---

- [**1. K-Nearest Neighbors**](#knn)
- [**2. Linear Regression**](#lr)
- [**3. Logistic Regression**](#lgr)
- [**4. Decision Tree**](#dt)
- [**5. Random Forest**](#rf)
- [**6. Gradient Boosting**](#gb)
- [**7. eXtreme Gradient Boosting**](#xg)

<a name="knn"></a>
#### **K-Nearest Neighbors**

At training time the KNN algorithm just memorizes the dataset which is $\mathcal{O}(1)$. Now consider at prediction time. Lets assume we have a dataset with $n$ datapoints living in a $m$-dimensional space. Consider a datapoint $x_0$. To determine the $k$ nearest neighbors of $x_0$, we calculate the distances from all datapoints to $x_0$. This is an $\mathcal{O}(mn)$. Then we select the $k$ datapoints that are closest to $x_0$, which requires $\mathcal{O}(kn)$ calculations. So the total time complexity  is $\mathcal{O}(mn+kn)$.  


<a name="lr"></a>
#### **Linear regression**

In linear regression we fit each sample $i$ with covariates $x_{ij}$ to the targets $y_i$ using the relationship:

$$y_i = \sum_j a_{j}x_{ij} + b $$

We have $i=1\ldots n$ and $j=1\ldots m$. One redefine the intercept $b$ as a parameter $a$ by defining $a\equiv (a,b)$ and $x\equiv (x,1)$. 

The exact solution is obtained by minimizing the mean square error:

$$a= (x^Tx)^{-1}x^Ty $$

To build the matrix $x^Tx$ it takes $\mathcal{O}(nm^2)$ operations, while the invertion is of order $\mathcal{O}(m^3)$. So in total $(x^Tx)^{-1}$ takes 
$\mathcal{O}(m^3+nm^2)$ operations. The resulting matrix is then multiplied by $x^Ty$ which is an operation of $\mathcal{O}(nm)$. Therefore we conclude that for fixed $m$ but $n\gg 1$, calculating the regression coefficients $a$ is a $\mathcal{O}(n)$ operation. However, if the number of features is scaled with $n$ fixed, then this an operation of order $\mathcal{O}(m^3)$.

At the time of prediction (a single datapoint), this is an operation of order $\mathcal{O}(m)$.

Other algorithms consist of gradient descent. In addition to the size of the dataset we have to account for the number of iterations or epochs $k$. We need to calculate the loss function which is an operation of order $\mathcal{O}(nm)$. And then calculate the derivatives with respect to the regression coefficients. This gives an order $\mathcal{O}(nm)$ calculation that we need to carry $k$ times. So in total we have $\mathcal{O}(nmk)$. This is still of order $n$ but linear in $m$ which is advantageous for high-dimensional covariates.

<a name="lgr"></a>
#### **Logistic Regression**

In the logistic regression model we assume a probability function of the form:

$$p(y=1|x_{ij}) = \frac{1}{1+\exp{(\sum_j w_j x_{ij} +b)}}$$

We train the model using maximum likelihood estimation. The optimization problem is convex which means we can use Newton-Raphson's method. In this method the loss function $L(w)$ (negative of log-likelihood) is approximated to quadratic order:

$$L(w^0+\Delta w)  =L(w^0) + \frac{\partial L}{\partial w_j} \Delta w_j + \frac{1}{2}\frac{\partial^2 L}{\partial w_j \partial w_i} \Delta w_j \Delta w_i$$

Then the minimum lies at:

$$\Delta w_i = -\left(\frac{\partial^2 L}{\partial w_j \partial w_i}\right)^{-1}\frac{\partial L}{\partial w_j} $$

We perform the descent using $w \rightarrow w +\Delta w$ iteratively. Both the first and second derivatives are of order  $\mathcal{O}(nm)$. Inverting the second derivative matrix, which is an $m\times m$ matrix, takes $\mathcal{O}(m^3)$. The matrix times vector multiplication takes $\mathcal{O}(m^3)$ too. The update of $w$ is $\mathcal{O}(m)$ process. So in total, and assuming that the number of iterations is small, this is an $\mathcal{O}(m^3+nm)$ calculation. So for a very large number of features this algorithm can become very expensive.

We can also consider gradient descent. As in linear regression, this is a $\mathcal{O}(nmk)$ calculation, with $k$ the number of steps. However, the gradient descent introduces an additional parameter, the learning rate and so we need to consider multiple experiments to find the optimal value.

<a name="dt"></a>
#### **Decision Tree**

The training of a decision tree consists in searching for the split that provides the highest information gain. For each feature, we sort their values and then loop over these to find the optimal split. Hence this takes $\mathcal{O}(n\log(n))$, due to sorting, and then $\mathcal{O}(n)$ for searching (calculating the information gain is also of order $\mathcal{O}(n)$). Looping over all features takes $\mathcal{O}(mn\log(n))$ at each level. 

The crucial aspect is that we only need to sort each feature once. After the first split is made the values remain sorted. Suppose the first split leads to a left dataset of size $n_l$ and a right dataset of size $n_r$ with $n=n_l+n_r$. Since we don't need to sort the data, searching for the best split takes $\mathcal{O}(mn_l)$ from the left and $\mathcal{O}(mn_r)$ from the right, which gives in total $\mathcal{O}(mn)$. Hence at each level we always have an $\mathcal{O}(mn)$ calculation. Since the number of levels $\sim \log(n)$ we have that the total complexity is

$$\mathcal{O}(mn\log(n))$$

For test complexity (one sample) we need to go down the tree so this requires $\mathcal{O}(\text{depth})\sim \mathcal{O}(\log(n))$.

<a name="rf"></a>
#### **Random Forest**

The random forest is an ensemble of decision trees, with each tree being trained on a bootstrap sample (with replacement) of the dataset. In building each tree, a random subsample of the features is considered in each split, to reduce statistical dependence between the trees. Sampling is an $\mathcal{O}(n)$ process, so in the worst case scenario a random forest with $k$ trees has total complexity, the complexity of training $k$ decision trees separately, that is, 

$$\mathcal{O}(kmn\log(n))$$

For test complexity, its just $k$ times the complexity of a single decision tree at prediction time or $\mathcal{O}(k\log(n))$.

<a name="gb"></a>
#### **Gradient Boosting**

In boosting, we fit sequentially shallow trees. Assume that we have $k$ trees of depth $d$, hence the total complexity is 

$$\mathcal{O}(mn\log(n)k)$$

Recall that building the tree along depth always gives a subleading term $\mathcal{O}(dmn)$, which is why the depth is not relevant here. At prediction time we have $\mathcal{O}(dk)$.

<a name="xg"></a>
#### **eXtreme Gradient Boosting**

The XGBoost algorithm uses a block structure and approximate spliting algorithms that improve the time complexity of gradient boosting. In a nutshell, features are sorted in blocks (partitions of the dataset) that allow efficient and parallelized computations. The approximate split algorithm consists in spliting the data according to quantiles rather then searching for the exact split. First a certain number of quantiles is calculated and the algorithms chooses one until that a certain accuracy is attained (this is controlled by $\epsilon$ hyperparameter). With the approximate split algorithm the overall time complexity is 

$$\mathcal{O}(kd|x|\log(q))$$

where $|x|$ 
denotes the number of non-missing entries in the data and $q$ the number of quantile candidates (usually $\sim 100$). If the data does not have missing values then 
$|x|=mn$.

With the block structure together with parallel processing we can reduce this time complexity to

$$\mathcal{O}(kd|x| + |x|\log(B))$$

where $B$ is the maximum number of rows in each block.