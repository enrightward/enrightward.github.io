---
title: "Order statistics — Part 3: Proof of expectation formula for uniform RVs"
date: 2021-03-08
categories: [statistics, order statistics]
tags: [statistics, order statistics, uniform distribution, expectation formula, gamma function, beta function]
preview_image: /enrightward.github.io/assets/img/order-statistics/part3/gamma.png
---

![Desktop View](/assets/img/order-statistics/part3/gamma.png)

## 1. Introduction

In the [last post](https://enrightward.github.io/enrightward.github.io/posts/order-statistics-part-2/), we proved the following general general formulae, after conducting some numerical experiments to gain intuition:

\begin{align}
F_{X_{(k)}}(x) &= \sum_{j=k}^{n} \binom{n}{j} F_{X}(x)^{j} (1 - F_{X}(x))^{n-j}, \newline
f_{X_{(k)}}(x) &= k \binom{n}{k} f_{X}(x) \cdot F_{X}(x)^{k-1} \, (1 - F_{X}(x))^{n-k}, \newline
\mathbb{E} \left \lbrack X_{(k)} \right \rbrack &=
k \binom{n}{k} \int_{0}^{1} x \cdot f_{X}(x) \cdot F_{X}(x)^{k-1} \, (1 - F_{X}(x))^{n-k} \, dx,
\end{align}

Note they depend only on $$n, k, f_{X}$$ and $$F_{X}$$. We will now round out the exposition of $$k^{\textrm{th}}$$ order statistics by specialising these general formulae to the case of the uniform distribution $$U(0, 1)$$. We will show that:

\begin{align}
F_{X_{(k)}}(x) &= \sum_{j=k}^{n} \binom{n}{j} x^{j} (1 - x)^{n-j}, \newline
f_{X_{(k)}}(x) &= k \binom{n}{k} x^{k-1} \, (1 - x)^{n-k}, \newline
\mathbb{E} \left \lbrack X_{(k)} \right \rbrack &= \frac{k}{n+1} \cdot
\end{align}

This post is purely theoretical, and has no code. The mathematical tools we use are the beta and gamma functions, introduced below.

## 2. Specialising the formulae to $$U(0, 1)$$

In the case where the $$X \sim U(0, 1)$$, we have $$F_{X}(x) = x$$ and $$f_{X}(x) = 1$$ on the unit interval, so the formulae computed above specialise to:

\begin{align}
F_{X_{(k)}}(x) &= \sum_{j=k}^{n} \binom{n}{j} x^{j}(1 - x)^{n - j}, \newline
f_{X_{(k)}}(x) &= k \binom{n}{k} x^{k-1} (1 - x)^{n-k}, \newline
\mathbb{E}[X_{(k)}] &= 
k \binom{n}{k} \int_{0}^{1} x \cdot x^{k-1} \, (1 - x)^{n-k} \, dx =
k \binom{n}{k} \int_{0}^{1} x^{k} (1 - x)^{n-k} \, dx.
\end{align}

## 3. Proving the expectation formulae using beta and gamma functions

The integral in the above expectation formula is an example of the [_beta function_](https://en.wikipedia.org/wiki/Beta_function), defined as

\begin{align}
B(z, w) := \int_{0}^{1} x^{z-1} (1 - x)^{w-1} \, dx.
\end{align}

This formula makes sense for complex numbers $$w$$ and $$z$$ with positive real part, but we are only interested in integer-valued arguments, because our exponents for $$x$$ and $$1-x$$ are $$k$$ and $$n-k$$, respectively. In the new notation of beta functions, our expectations are written:

\begin{align}
\mathbb{E}[X_{(k)}] = k \binom{n}{k} \int_{0}^{1} x^{k} (1 - x)^{n-k} \, dx = k \binom{n}{k} B(k+1, n-k+1).
\end{align}

This is interesting to us because the beta function can be computed in terms of the [_gamma function_](https://en.wikipedia.org/wiki/Gamma_function)

\begin{align}
\Gamma(z) := \int_{0}^{\infty} x^{z-1} e^{-x} \, dx,
\end{align}

and the gamma function is a generalisation to the (positive half) complex plane of the factorial function $$n \mapsto n!$$, hence reduces to factorial computations for integer-valued arguments. Concretely, the relations we need are:

\begin{align}
B(z, w) &= \frac{\Gamma(z) \Gamma(w)}{\Gamma(z + w)}, \quad \textrm{and} \newline
\Gamma(n) &= (n-1)!
\end{align}

We will show these relations shortly. For now, observe they imply the formulae we conjectured for the expectations of $$X_{(k)}$$. Indeed, using each of these results in turn, we have

\begin{align}
B(k+1, n-k+1) = \frac{\Gamma(k+1) \Gamma(n-k+1)}{\Gamma(n+2)} = 
\frac{k!(n-k)!}{(n+1)!} = \frac{1}{(n+1) \binom{n}{k}},
\end{align}

and hence

\begin{align}
\mathbb{E}[X_{(k)}] &= k \binom{n}{k} B(k+1, n-k+1) 
= \frac{k \binom{n}{k}}{(n+1) \binom{n}{k}} = \frac{k}{n+1} \cdot
\end{align}

## 4. Proving the necessary properties of beta and gamma functions

Now we prove the relations for $$B(z, w)$$ and $$\Gamma(z)$$. To see that $$\Gamma(n) = (n-1)!$$, it suffices to prove the recursive formula 

\begin{align}
\Gamma(z) = (z-1)\Gamma(z-1),
\end{align}

for any $$z$$ with positive real part. For then we would have:

\begin{align}
\Gamma(n) &= (n-1)\Gamma(n-1) \newline
&= (n-1)(n-2)\Gamma(n-2) \newline
\vdots \newline
&= (n-1)(n-2)(n-3) \ldots (2)(1) = (n-1)!
\end{align}

We show this using the definite integral version of the integration by parts formula:

\begin{align}
\int_{a}^{b} u \, dv = \left \lbrack uv \right \rbrack_{a}^{b} - \int_{a}^{b} v \, du, 
\end{align}

with $$u = x^{z-1}$$ and $$dv = e^{-x} dx$$, implying that $$du = (z-1)x^{z-2} dx$$ and $$v = -e^{-x}$$. Note this formula holds also in the case where $$a$$ or $$b$$ is $$\pm \infty$$, provided we interpret the term $$\left \lbrack uv \right \rbrack_{a}^{b}$$ as expressing the appropriate limit(s). We have:

\begin{align}
\Gamma(z) &= \int_{0}^{\infty} x^{z-1} e^{-x} \, dx \newline
&= \lim_{b \rightarrow \infty} \left \lbrack -x^{z-1} e^{-x} \right \rbrack_{0}^{b} + (z-1) \int_{0}^{\infty} x^{z-2} e^{-x} \, dx \newline
&= 0 + (z-1) \int_{0}^{\infty} x^{z-2} e^{-x} \, dx \newline
&= (z-1)\Gamma(z-1).
\end{align}

Now we prove the formula connecting $$B(z, w)$$ and $$\Gamma(z)$$. I'm taking this proof [from Wikipedia](https://en.wikipedia.org/wiki/Beta_function#Relationship_to_the_gamma_function) and filling in details on the variable substitution using [these lecture notes](https://homepage.tudelft.nl/11r49/documents/wi4006/gammabeta.pdf) (Theorem 2, page 3). Note: Curiously, although the Wikipedia article cites Emil Artin's book [Gamma Functions](https://web.archive.org/web/20161112081854/http://www.plouffe.fr/simon/math/Artin%20E.%20The%20Gamma%20Function%20(1931)(23s).pdf), pages 18-19, Artin's proof is not the one written down there.

In any case, the argument begins like this:

\begin{align}
\Gamma(z) \, \Gamma(w) &= \int_{x=0}^{\infty} x^{z-1} e^{-x} \, dx \int_{y=0}^{\infty} y^{w-1} e^{-y} \, dy \newline
&= \int_{x=0}^{\infty} \int_{y=0}^{\infty} e^{-(x+y)} x^{z-1} y^{w-1} \, dx \, dy.
\end{align}

Now we introduce an implicit change of variables $$s, t$$, defined by the conditions $$x = st$$ and $$y=s(1 - t)$$. This transformation has Jacobian matrix:

\begin{align}
J(x(s, t), y(s, t)) = \left( 
\begin{array}{cc}
\frac{\partial x}{\partial s} & \frac{\partial x}{\partial t} \newline
\frac{\partial y}{\partial s} & \frac{\partial y}{\partial t}
\end{array} \right) = \left( 
\begin{array}{cc}
t & s \newline
1-t & -s
\end{array} \right),
\end{align}

with determinant $$\det(J) = -st - s(1-t) = -s$$. It follows that 

\begin{align}
dx \, dy = \left\| \det{J} \right\| ds \, dt = s \, ds \, dt. 
\end{align}

Observe that since $$x$$ and $$y$$ range over $$[0, \infty)$$ and $$x + y = st + s(1-t) = s$$, then $$s$$ must range over 
$$[0, \infty)$$, too. On the other hand,

\begin{align}
t = \frac{x}{s} = \frac{x}{x + y},
\end{align}

so $$t$$ ranges only over the unit interval $$[0, 1]$$. Making these substitutions gives:

\begin{align}
\Gamma(z) \cdot \Gamma(w) &=
\int_{x=0}^{\infty} \int_{y=0}^{\infty} e^{-(x+y)} x^{z-1} y^{w-1} \, dx \, dy \newline
&= \int_{s=0}^{\infty} \int_{t=0}^{1} e^{-s} (st)^{z-1} (s(1 - t))^{w-1} \, s \, dt \, ds \newline
&= \int_{s=0}^{\infty} \int_{t=0}^{1} e^{-s} s^{z+w-1} t^{z-1}(1 - t)^{w-1} \, dt \, ds \newline
&= \int_{0}^{\infty} s^{z+w-1} e^{-s} \left\\{ \int_{0}^{1} t^{z-1}(1 - t)^{w-1} \, dt \, \right\\} ds \newline
&= B(z, w) \int_{0}^{\infty} s^{z+w-1} e^{-s} ds \newline
&= B(z, w) \cdot \Gamma(z+w).
\end{align}

Dividing both sides of this equation by $$\Gamma(z+w)$$ now gives the result.

## 5. Roundup

We specialised the general formulae:

\begin{align}
F_{X_{(k)}}(x) &= \sum_{j=k}^{n} \binom{n}{j} F_{X}(x)^{j} (1 - F_{X}(x))^{n-j}, \newline
f_{X_{(k)}}(x) &= k \binom{n}{k} f_{X}(x) \cdot F_{X}(x)^{k-1} \, (1 - F_{X}(x))^{n-k}, \newline
\mathbb{E} \left \lbrack X_{(k)} \right \rbrack &=
k \binom{n}{k} \int_{0}^{1} x \cdot f_{X}(x) \cdot F_{X}(x)^{k-1} \, (1 - F_{X}(x))^{n-k} \, dx,
\end{align}

which depend only on $$n, k, f_{X}$$ and $$F_{X}$$, to the case of the uniform distribution $$U(0, 1)$$. We showed that:

\begin{align}
F_{X_{(k)}}(x) &= \sum_{j=k}^{n} \binom{n}{j} x^{j} (1 - x)^{n-j}, \newline
f_{X_{(k)}}(x) &= k \binom{n}{k} x^{k-1} \, (1 - x)^{n-k}, \newline
\mathbb{E} \left \lbrack X_{(k)} \right \rbrack &= \frac{k}{n+1} \cdot
\end{align}

To prove the expectation formula, we introduced the beta and gamma function, and proved some identities they satisfy.
