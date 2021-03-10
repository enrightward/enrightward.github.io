---
title: Cross entropy — Part 2
date: 2021-02-24
categories: [statistics, entropy]
tags: [statistics, cross entropy, softmax, convexity]
preview_image: /enrightward.github.io/assets/img/cross-entropy/convex.png
---

![Desktop View](/assets/img/cross-entropy/convex.png)

## Introduction

In the previous post, I recalled the definition of the _cross entropy_ $$H(p, q)$$ of two discrete PDFs $$p$$ and $$q$$ over the same support $$S = \{ x_{1}, \ldots, x_{n} \}$$. It is a loose measure of similarity of $$p$$ and $$q$$, and so is used in machine learning to define objective functions for tasks where the goal is to learn a PDF $$p$$ implicit in training data by updating the internal parameters of a learnt PDF $$q$$. I also wrote down a proof that for fixed but arbitrary $$p$$, the function $$h_{p}: q \mapsto H(p, q)$$ obtains a global minimum at $$q = p$$.

In this post, I will write down a technical result that $$h_{p}(q)$$ is a convex function, provided we restrict $$p$$ and $$q$$ to the space of all PDFs such that no event in $$S$$ is impossible, i.e. we assume the $$p(x_{i})$$ and $$q(x_{i})$$ are never zero. This assumption is not very restrictive for machine learning purposes, where learnt zero probabilities are rare. It also allows us to use the _softmax_ parametrisation for $$q$$, explained below.

## Cross entropy

Recall that the _cross entropy_ of a pair $$(p, q)$$ of discrete PDFs with the same support $$S$$ is defined to be:

\begin{equation}
H(p, q) := -\sum_{x \in S}^{n} p(x) \log(q(x)).
\end{equation}

## The softmax parametrisation of a discrete PDF

For any collection $$t_{1}, \ldots, t_{n}$$ of real numbers, the $$n$$-tuple:

\begin{equation}
q(t_{1}, \ldots, t_{n}) := \left( \frac{e^{t_{1}}}{Z}, \ldots, \frac{e^{t_{n}}}{Z} \right),
\end{equation}

where $$Z$$ is the normalisation constant $$\sum_{i=1}^{n} e^{t_{i}}$$, is a PDF. Indeed, each entry of $$q$$ is positive-valued, since its numerator and denominator are sums of real-valued exponentials, and these entries sum to $$1$$, by construction of $$Z$$. Conversely, any PDF $$q = (q_{1}, \ldots, q_{n})$$ with all non-zero entries can be written in this form: Simply set $$t_{i}$$ equal to $$\log(q_{i})$$. We call this the _softmax parametrisation_ of $$q$$.

## Proof that $$p \mapsto H(p, q)$$ is convex

We showed in the previous post that the function $$q \mapsto H(p, q)$$, with $$p$$ fixed and $$q$$ varying, has a global minimum at $$q = p$$. We now show that this is a convex function, assuming $$p$$ and $$q$$ have no zero entries. By the above assumption, we can write $$q$$ using the softmax parametrisation:

\begin{equation}
q(x_{j}) = \frac{e^{t_{j}}}{Z},
\end{equation}

where $$Z$$ is the normalisation constant $$\sum_{i=1}^{n} e^{t_{i}}$$.

## Step 1 -- softmax parametrisation:
Fix $$p$$ and re-write $$H(p, q)$$ by replacing $$q(x_{i})$$ with $$e^{t_{i}}/Z$$:

\begin{equation}
H(p, q) = -\sum_{i=1}^{n} p(x_{i}) \log(q(x_{i})) = -\sum_{i=1}^{n} p(x_{i}) \log \left( \frac{e^{t_{i}}}{Z} \right) = 
\sum_{i=1}^{n} p(x_{i})(\log(Z) - t_{i}) = \log(Z) - \sum_{i=1}^{n} p(x_{i}) t_{i}.
\end{equation}

## Step 2 -- first derivative:
Now we find a local optimum for $$H(p, q)$$, regarded as a function of $$q$$ for fixed but arbitrary $$p$$. The first step is to solve $$\partial H/\partial t_{j} = 0$$. To do this, observe that because $$Z = \sum_{i=1}^{n} e^{t_{i}}$$, we have $$\partial Z/\partial t_{j} = e^{t_{j}}$$, so that:

\begin{equation}
\frac{\partial \log(Z)}{t_{j}} = \frac{e^{t_{j}}}{Z} = q(x_{j}),
\end{equation}

and hence:

\begin{equation}
\frac{\partial H(p, q)}{\partial t_{j}} = q(x_{j}) - p(x_{j}).
\end{equation}

This partial derivative is zero exactly when $$q(x_{j}) = p(x_{j})$$.

## Step 3 -- second derivative:
The previous step implies that for fixed $$p$$, the quantity $$H(p, q)$$ is locally flat around $$q = p$$. To show this is a global minimum, we will show $$H$$ is "concave up" as a function in $$q$$. It suffices to show that the Hessian matrix:

\begin{equation}
\nabla^{2} H(p, q) = \left( \frac{\partial^{2} H(p, q)}{\partial t_{j} \partial t_{i}} \right)_{i, j=1}^{n}
\end{equation}

is positive definite, i.e. satisfies $$v^{t} \nabla^{2} H(p, q) v \ge 0$$ for all $$v \in \mathbb{R}^{n}$$, with equality only for $$v = 0$$. First let's compute the entries of the Hessian. From the partial derivative calculation in the previous step, we have:

\begin{equation}
\frac{\partial^{2} H(p, q)}{\partial t_{j} \partial t_{i}} = \frac{\partial}{\partial t_{j}}(q(x_{i}) - p(x_{i})) = 
\frac{\partial q(x_{i})}{\partial t_{j}},
\end{equation}

because $$p$$ is constant. Now, 

\begin{equation}
\frac{\partial q(x_{i})}{\partial t_{j}} = \frac{\partial}{\partial t_{j}} \left( \frac{e^{t_{i}}}{Z} \right) = 
\frac{Z \cdot \frac{\partial e^{t_{i}}}{\partial t_{j}} - e^{t_{i}} \cdot \frac{\partial Z}{\partial t_{j}}}{Z^{2}} = 
\frac{Z \cdot \frac{\partial e^{t_{i}}}{\partial t_{j}} - e^{t_{i} + t_{j}}}{Z^{2}},
\end{equation}
noting that $$\frac{\partial Z}{\partial t_{j}} = e^{t_{j}}$$ from Step 2. On the other hand, the quantity $$\frac{\partial e^{t_{i}}}{\partial t_{j}}$$ is equal either to zero, if $$i$$ and $$j$$ differ, or else $$e^{t_{i}}$$, if $$i$$ and $$j$$ are the same. It follows that:

\begin{equation}
\frac{\partial^{2} H(p, q)}{\partial t_{j} \partial t_{i}} = -\frac{e^{t_{i} + t_{j}}}{Z^{2}} = -q(x_{i})q(x_{j}),
\end{equation}

if $$i \neq j$$, and

\begin{equation}
\frac{\partial^{2} H(p, q)}{\partial t_{i}^2{}} = \frac{Z e^{t_{i}} - e^{2t_{i}}}{Z^{2}} = q(x_{i}) - q(x_{i})^{2}.
\end{equation}

The Hessian can thus be written:

\begin{equation}
\nabla^{2} H(p, q) = \textrm{diag}(Q) - QQ^{t},
\end{equation}

where $$Q = (q(x_{1}), \ldots, q(x_{n}))$$ is an $$n$$-dimensional column vector, and $$\textrm{diag}(Q)$$ is the $$n \times n$$ matrix whose diagonal is defined by $$Q$$, and whose off-diagonal entries are zero. Observe now that for $$v \in \mathbb{R}^{n}$$, we have:

\begin{equation}
v \nabla^{2} H(p, q) v^{t} = v \, \textrm{diag}(Q) v^{t} - v QQ^{t} v^{t} = \\
\sum_{i=1}^{n} v_{i}^{2} q(x_{i}) - \sum_{i=1}^{n} v_{i}^{2} q(x_{i})^{2} =
\sum_{i=1}^{n} v_{i}^{2} q(x_{i}) (1 - q(x_{i})).
\end{equation}
Here, each summand $$v_{i}^{2} q(x_{i}) (1 - q(x_{i}))$$ is non-negative, being the product of a square, a probability and its complementary probability. It follows that the whole sum is non-negative. Since $$p$$ and $$q$$ have no zero probabilities by assumption, this sum can be zero only if each $$v_{i}$$ is zero, so $$\nabla^{2} H(p, q)$$ is positive definite and the minimum $$q = p$$ is global.

## Why is this convexity important?

In the previous post, I noted that a common machine learning task is to update a parametrised PDF $$q$$ to more closely resemble an idealised PDF $$p$$ implicit in some data set, and illustrated this task with the example of language modelling. The above convexity result, combined with last post's proof that $$q \mapsto H(p, q)$$ achieves a minimum at $$q = p$$, implies this optimisation is achievable with gradient descent.
