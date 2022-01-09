---
layout: post
title: "Hoeffding's inequality"
date: 2020-05-05
category: Statistics
excerpt: We derive Hoeffding's inequality. This is one of the most used results in machine learning theory.
katex: True
---

### **Hoeffding's inequality**
<br/>
Let $X_1,\ldots,X_m$ be $m$ independent random variables (not necessarily identically distributed). All $X_i$ takes values in $[a_i,b_i]$. Then for any $\epsilon>0$ we have

$$\mathbb{P}(|S_m-E(S_m)|\geq\epsilon)\leq e^{-2\epsilon^2/\sum_i(b_i-a_i)^2},\;S_m=\sum_{i=1}^mX_i$$

If we have $a_i=a_j=a$ and $b_i=b_j=b$ for $\forall i,j$ then we have a version of the Hoeffding's inequality which is most known

$$\mathbb{P}(|\hat{X}_m-E(\hat{X}_m)|\geq\epsilon)\leq e^{-2m\epsilon^2/(b-a)^2},\; \hat{X}_m=\frac{1}{m}\sum_{i=1}^mX_i$$

First we show that for $t>0$ we have

$$\begin{equation}\begin{split}\mathbb{P}(x\geq y)\leq e^{-ty}E(e^{t x})\end{split}\end{equation}$$

Note that

$$e^{-ty}E(e^{tx})=\sum_{x\in X}e^{t(x-y)}P(x)$$

with $\sum_{x\in X}P(x)=1$. We expand the r.h.s as

$$\begin{equation*}\begin{split}\sum_{x\in X}e^{t(x-y)}P(x)&=\sum_{x\geq y}e^{t(x-y)}P(x)+\sum_{x<y}e^{t(x-y)}P(x)\\
&\geq \sum_{x\geq y}e^{t(x-y)}P(x)\\
&\geq  \sum_{x\geq y}e^{t(x-y)}P(x)=\sum_{x\geq y}P(x)=P(x\geq y)\end{split}\end{equation*}$$

Then we use the auxiliary distribution $P'(a)=(b-x)/(b-a)$ and $P'(b)=(x-a)/(b-a)$ with $a\leq x\leq b$ and $P'(a)+P'(b)=1$, to show that

$$e^{tx}\leq \frac{b-x}{b-a}e^{ta}+\frac{x-a}{b-a}e^{tb}$$

because of the convexity of $e^{tx}$. Assuming that $E(x)=0$ (this implies that $a<0$ and $b>0$), we take the average on $x$ on both sides of the above equation to get

$$E(e^{tx})\leq \frac{b}{b-a}e^{ta}-\frac{a}{b-a}e^{tb}=\frac{e^{\phi(t)}}{b-a}$$

with $\phi(t)=\ln(be^{ta}-ae^{tb})$. We can show that $\phi(t)$ is a convex function of $t$ with $\phi''(t)\leq (b-a)^2/4$ (essentially we need to show that $\phi''(t)$ has a maximum equal to $(b-a)^2/4$). Using that $\phi'(t=0)=0$ we also have $\phi'(t)\leq (b-a)^2t/4$. Then integrating again we have $\phi(t)\leq \phi(0)+(b-a)^2t^2/8$. This gives us

$$\begin{equation}\begin{split}E(e^{tx})\leq e^{t^2(b-a)^2/8}\end{split}\end{equation}$$

Using inequalities Eq.1 and Eq.2, we calculate

$$\begin{equation*}\begin{split}P(\hat{X}_m-E(\hat{X}_m)>\epsilon)&\leq e^{-t\epsilon}E(e^{t(\hat{X}_m-E(\hat{X}_m))})\\
&=e^{-t\epsilon}\prod_iE(e^{t(X_i-E(X))})\\
&\leq e^{-t\epsilon} e^{t^2\sum_i(b_i-a_i)^2/8}\end{split}\end{equation*}$$

We can choose $t$ such that the bound is optimal (this corresponds to the minimum of the exponent). We obtain

$$P(\hat{X}_m-E(\hat{X}_m)>\epsilon)\leq e^{-2\epsilon^2/\sum_i(b_i-a_i)^2}$$
