---
layout: post
title: "Rademacher complexity"
date: 2020-05-2
category: Machine Learning
excerpt: The Rademacher complexity measures how a hypothesis correlates with noise. This gives a way to evaluate the capacity or complexity of a hypothesis class.
katex: True
---

### **Table of contents**

1. [Definition](#def)
2. [Bounds](#bounds)

<a name="def"></a>
### **1. Definition**

The empirical Rademacher complexity of a hypothesis class $G=\{g\}$ is defined as an average over the training set $S=(z_1,\ldots,z_m)$ in the following way:

$$\hat{\mathcal{R}}(G)=E_{\sigma}\left[\text{sup}_{g\in G}\frac{1}{m}\sum_{i=1}^{m}\sigma_i g(z_i)\right]$$

where $\sigma_i$ are $m$ independently and uniformly distributed random variables in the interval $[-1,1]$. Since $E(\sigma)=0$, we see that the average above is the correlation between $\sigma$ and $g(z)$. The Rademacher complexity, therefore, measures how well a hypothesis class correlates with noise. If a class has enough complexity, it will correlate more easily with noise and have higher Rademacher complexity.

The Rademacher complexity, rather than the empirical one, is in turn defined as the statistical average over the true distribution $D(z)^m$ on all the possible sets of size $m$:

$$\mathcal{R}_m(G)=E_{\sim D^m}(\hat{\mathcal{R}}(G))$$

Note that the expression above is explicitly dependent on $m$ because one cannot move the expectation in $z$ over to $g(z)$ inside the definition of the empirical Rademacher complexity.

For example, suppose we have a linear classifier in two dimensions 
$g(x\in \mathbb{R}^2)$
, which is a line that classifies points as $\{-1,1\}$ depending on whether the point is above or below the line. If we have up to three points, one can always choose a line that classifies all the points correctly. This is a consequence of the VC dimension of $\mathbb{R}^2$ being three. Then the above supremum is attained by picking a classifier $g$ such that 

$$\text{sup}_{g\in G} \sum_{i=1}^{m}\sigma_i g(z_i)=\sum_{i=1}^{m}|\sigma_i|$$

, which is always possible if we have up to three points. The Rademacher complexity is simply 

$$\mathcal{R}_{m\leq 3}=E_{\sigma}|\sigma|$$

, and thus independent of $m$. The same follows in higher dimensions. The Rademacher complexity is independent of $m$ if $m$ is less than the VC dimension. For $m$ bigger than the VC dimension, we can find the following bound. 

<a name="bounds"></a>
### **2. Bounds**

One can determine several bounds on the Rademacher complexity. One of particular interest takes into account the growth function. Remember that the growth function $\Pi(m)$ is the maximal number of distinct ways of classifying a set of $m$ points $z_1,\ldots,z_m$ using an hypothesis class $\mathcal{H}$. In order to calculate this bound we need the following lemma:

*Massart's Lemma: let $A\subset \mathbb{R}^m$ be a finite set, and $r=\text{max}_{x\in A}\|x\|_2$, then*

$$E_{\sigma}\left[\frac{1}{m}\text{sup}_{x\in A}\sum_{i=1}^{m} \sigma_ix_i\right]\leq \frac{r\sqrt{2\ln|A|}}{m}$$

where $\sigma_i$ are independent and uniformly distributed random variables in the interval $[-1,1]$. The proof goes by first using Jensen's inequality:

$$\begin{equation*}\begin{split}\exp(t E_{\sigma}[\text{sup}_{x\in A}\sum_{i=1}^{m} \sigma_i x_i])\leq E_{\sigma}\exp(t\text{sup}_{x\in A}\sum_{i=1}^m \sigma_i x_i)\end{split}\end{equation*}$$

Now since the exponential function is monotically increasing we have that:

$$E_{\sigma}\exp(t\text{sup}_{x\in A}\sum_{i=1}^m \sigma_i x_i)=E_{\sigma}\text{sup}_{x\in A}\exp(t\sum_{i=1}^m \sigma_i x_i)\leq \sum_{x\in A} E_{\sigma}\exp(t\sum_{i=1}^m \sigma_i x_i)$$

Next we use the inequality nr. 2 from Hoeffding's inequality post which states that for a random variable $w\in [a,b]$ with $E(w)=0$ we have:
$$E_w\exp(tw)\leq \exp(t^2(b-a)^2/8)$$

This means that:

$$\sum_{x\in A} E_{\sigma}\exp(t\sum_{i=1}^m \sigma_i x_i)=\sum_{x\in A}\prod_iE_{\sigma_i}\exp(t \sigma_i x_i)\leq \sum_{x\in A} \exp(t^2x_i^2/2)\leq |A| \exp(t^2 r^2/2)$$

where $|A|$ is the "size" of the set $A$ and 
$r^2=\text{max}_{x\in A}\|x\|_2$
Using this result in Eq.1 and taking the log on both sides of the inequality:

$$E_{\sigma}[\text{sup}_{x\in A}\sum_{i=1}^{m} \sigma_i x_i]\leq \frac{\ln|A|}{t}+\frac{r^2}{2}t$$

The optimal bound corresponds to 

$$t=\sqrt{2\ln|A|/r^2}$$

which is the value where the function on the right side obtains its minimum. The final result is:

$$E_{\sigma}[\text{sup}_{x\in A}\sum_{i=1}^{m} \sigma_i x_i]\leq r\sqrt{2\ln |A|}$$

We can apply this result to determine a bound on the Rademacher complexity for hypothesis classes with target $\{-1,1\}$. So we have

$$E_{D^m(z)}E_{\sigma}\left[\text{sup}_{g\in G}\frac{1}{m}\sum_{i=1}^{m}\sigma_i g(z_i)\right]\leq E_{D^m(z)}\frac{r}{m}\sqrt{2\ln |A|}$$

We can easily calculate 
$r^2=\sum_i^mx_i^2=m$
 and thus $r=\sqrt{m}$. Moreover we know that, by definition, $|A|\leq \Pi(m)$, the growth function, and hence we find:

$$\mathcal{R}_m\leq \sqrt{\frac{2\ln \Pi(m)}{m}}$$