---
layout: post
title: "Random Forest"
date: 2020-09-13
category: Machine Learning
image: randomforest.png
excerpt: A random forest is an ensemble of decision trees. The trees are fitted in random samples of the training set, preventing overfitting and reducing variance. 
katex: True
---

- [**1. Bagging and Decision Trees**](#1-bagging-and-decision-trees)
- [**2. Ensembles and Random forest**](#2-ensembles-and-random-forest)
- [**3. Python Implementation**](#3-python-implementation)

<a name="def1"></a>
### **1. Bagging and Decision Trees**

Bagging, short for bootstrap aggregating, is the process by which we train an ensemble of machine learning models using datasets sampled from the empirical distribution. This process helps reduce variance and overfitting. Given a dataset $S$, we generate $m$ samples $S'$ of size $n$, by drawing datapoints from $S$ uniformly and with replacement. We can then create an ensemble by fitting $m$ models on each sample and averaging (regression) or voting (classification) the result of each of the models. If each of these models is a decision tree, then this ensemble is a random forest.

To take advantage of the bootstrapping mechanism, each of the ensemble models must be independent of each other. This is not always the case because usually there are features the model learns more strongly than others, effectively making the different models depend on each other. To remediate this, we do not allow the decision tree to learn all the features. Instead, each of the models knows different subsets of features.  After fitting the models, the predicted class is determined by the majority vote. In the case of regression, we average each of the predictions. 

<a name="forest"></a>
### **2. Ensembles and Random forest**

We analyze the effect of bootstrapping decision trees on the generalization error and bias/variance tradeoff. 

Suppose we have $m$ models $V^{a}$, with $a=1\ldots m$. In the case of regression, consider the model average 

$$\bar{V}(x)=\sum_a \omega_a V^a(x)$$

where $\omega_a$ are some weights. The ambiguity $A(x)^a$ for the model $a$ is defined as 

$$A^a(x)=(V^a(x)-\bar{V}(x))^2$$

and the ensemble ambiguity $A(x)$ is obtained by taking the ensemble average

$$A(x)=\sum_a \omega_aA^a(x)=\sum_a \omega_a(V^a(x)-\bar{V}(x))^2$$

The error of a model and the ensemble, respectively $\epsilon^a$ and $\epsilon$, are

$$\begin{equation*}\begin{split}&\epsilon^a(x)=(y(x)-V^a(x))^2 \\
&\epsilon= (y(x)-\bar{V}(x))^2
\end{split}\end{equation*}$$

One can easily show that

$$A(x)=\sum_a \omega_a\epsilon^a(x)-\epsilon(x)=\bar{\epsilon}(x)-\epsilon(x)$$

where we defined the ensemble average $\bar{\epsilon}=\sum_a \omega_a\epsilon^a$. Averaging this quantities over the distribution of $x$, $D(x)$, we obtain an equation involving the generalization error of the ensemble and of the individual components, that is

$$E=\bar{E}-A$$

where $E=\int dx \epsilon(x) D(x)$ is the generalization error and $A=\int dx A(x) D(x)$ is the total ambiguity.

Note that the ambiguity $A$ only depends on the models $V^a$ and not on labeled data. It measures how the different models correlate with the average. Since $A$ is always positive, we can conclude that the generalization error is smaller than the average error. 

If the models are highly biased, we expect similar predictions across the ensemble, making $A$ small. In this case, the generalization error will be essentially the same as the average of the generalization errors. However, if the predictions vary a lot from one model to another, the ambiguity will be higher,  making the generalization smaller than the average. So we want the models to disagree! Random forests implement this by letting each decision tree learn on a different subset of every split feature. This results in a set of trees with different split structure: 

<div style="text-align: center"><img src="/images/randomforest.png"  width="80%"></div>

Another important aspect of ensemble methods is that they do not increase the bias of the model. For instance 

$$\begin{equation*}\begin{split}\text{Bias}=f(x)-\mathbb{E}\bar{V}(x)=\sum_a \omega_a (f(x)-\mathbb{E}V^a(x))=\sum_a \omega_a \text{Bias}^a=\text{bias}
\end{split}\end{equation*}$$

where $\text{bias}$ is the bias of an individual model, assuming that each model has similar bias. On the other hand, the variance 

$$\mathbb{E}(\bar{V}(x)-\mathbb{E}\bar{V}(x))^2=\sum_a \omega_a^2(V^a-\mathbb{E}V^a)^2+\sum_{a\neq b}\omega_a\omega_b(V^a-\mathbb{E}V^a)(V^b-\mathbb{E}V^b)$$

We do not expect the quantities $(V^a-\mathbb{E}V^a)^2$ and $(V^a-\mathbb{E}V^a)(V^b-\mathbb{E}V^b)$ to differ significantly across the models, and so defining

$$\text{Var}\equiv (V^a-\mathbb{E}V^a)^2,\; \rho(x)\equiv\frac{(V^a-\mathbb{E}V^a)(V^b-\mathbb{E}V^b)}{\text{Var}(x)}$$

we obtain

$$\begin{equation*}\begin{split}\mathbb{E}(\bar{V}(x)-\mathbb{E}\bar{V}(x))^2&=\text{Var}(x)\sum_a \omega_a^2 + \rho(x)\text{Var}(x) \sum_{a\neq b}\omega_a\omega_b\\
&=\text{Var}(x)(1-\rho(x))\sum_a\omega_a^2+\rho(x)\text{Var}(x)<\text{Var}(x)\end{split}\end{equation*}$$

This quantity has a lower bound at $\omega_a=1/m$, the uniform distribution. This means that

$$\text{Var}(x)\frac{(1-\rho(x))}{m}+\rho(x)\text{Var}(x)\leq \mathbb{E}(\bar{V}(x)-\mathbb{E}\bar{V}(x))^2\leq \text{Var}(x)$$

If the models are averaged with constant weights, then $\sum_a \omega_a^2$ tends to zero as $m\rightarrow \infty$, and the variance is the product of the correlation $\rho(x)$ and the individual model variance.


<a name="python"></a>
### **3. Python Implementation**

```python
class RandomForest:
    def __init__(self,n_estimators,params):
        self.n_estimators=n_estimators
        self.params=params
        self.trees=None
        
    def fit(self,x,y):
        n_instances=x.shape[0]
        self.classes=sorted(set(y))
        self.trees=[]
        for n in range(self.n_estimators):
            idx=np.arange(n_instances)
            idx_sample=np.random.choice(idx,n_instances,replace=True)
            xsample=x[idx_sample]
            ysample=y[idx_sample]
            tree=DecisionTreeClassifier(**self.params,max_features='auto')
            tree.fit(xsample,ysample)
            self.trees.append(tree)
            
    def predict(self,x):
        classes=self.trees[0].classes_
        dic={i:cl for i,cl in enumerate(classes)}
        ypred=self.trees[0].predict_proba(x)
        for tree in self.trees[1:]:
            ypred+=tree.predict_proba(x)
        ypred=ypred
        ypred=ypred.argmax(axis=1)
        ypred=np.vectorize(dic.get)(ypred)
        
        return ypred
```