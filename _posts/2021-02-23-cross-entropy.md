# Cross entropy$$$$

Suppose $$p$$ is a discrete PDF with support ("outcome space") $$S = \{ x_{1}, \ldots, x_{n} \}$$. The _entropy_ of $$p$$ is defined to be:

\begin{equation}
H(p) := \sum_{i=1}^{n} p(x_{i}) \log \left( \frac{1}{p(x_{i})} \right) = -\sum_{i=1}^{n} p(x_{i}) \log(p(x_{i})),
\end{equation}

provided none of the $$p(x_{i})$$ is zero. If $$p(x_{i}) = 0$$ for some $$i$$, then we declare by fiat that the summand $$p(x_{i}) \log(p(x_{i}))$$ is zero. One justification for this is that the limit of $$x \log x$$ is zero as $$x$$ approaches zero from the positive number line.

The quantity $$\log(1/p(x_{i}))$$ is called the _surprisal_ of the outcome $$x_{i}$$, and quantifies the information associated to $$x_{i}$$. If the logarithm is base 2, this information amount is in bits. Let $$q$$ be a second PDF with the same support $$S$$. Then the _cross entropy_ of the pair $$p$$ and $$q$$ is defined to be:

\begin{equation}
H(p, q) := -\sum_{i=1}^{n} p(x_{i}) \log(q(x_{i})).
\end{equation}

This is not a symmetric quantity: $$H(p, q)$$ and $$H(q, p)$$ are different in general. We'll see this by computing examples. To do this, we first define two python functions.


```python
import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(p, q):
    """Require that p and q are numpy arrays"""
    result = -p.dot(np.log(q))
    return result

def random_discrete_pdf(length):
    """Compute a random multinomial 
    PDF with `length` entries"""
    # Generate `length` random numbers between 0 and 1.
    pdf = np.random.rand(length)
    # normalise, so it's a pdf
    pdf = pdf/pdf.sum()
    return pdf
```

Taking $$p$$ and $$q$$ to be two randomly-generated PDFs of length five, we have:


```python
length = 5
p = random_discrete_pdf(length)
q = random_discrete_pdf(length)
print('p =', p)
print('q =', q)
print()
print('H(p, q) = {:0.3f}'.format(cross_entropy(p, q)))
print('H(q, p) = {:0.3f}'.format(cross_entropy(q, p)))
```

    p = [0.15357802 0.26044683 0.07609958 0.00201638 0.50785919]
    q = [0.04635283 0.32557158 0.21084165 0.39774773 0.01948621]
    
    H(p, q) = 2.884
    H(q, p) = 3.550


Now let's keep $$p$$ fixed, and compute $$H(p, q)$$ for randomly-generated $$q$$:


```python
N = 1000000
hpqs = []

for _ in range(N):
    q = random_discrete_pdf(length)
    hpq = cross_entropy(p, q)
    hpqs.append(hpq)
    
hpqs = np.array(hpqs)
```

We plot the $$H(p, q)$$'s in a histogram, to get a sense of the distribution.


```python
num_bins = 1000
plt.subplots(figsize=(12, 5))
n, bins, patches = plt.hist(hpqs, num_bins, density=1, facecolor='blue', alpha=0.5)

plt.xlabel('H(p, q)')
plt.ylabel('Frequency')
plt.title(r'Distribution of H(p, q)')
plt.show()
```

![png](/assets/img/hpq_histogram.png)

There appears to be a hard cut-off around $$H(p, q) = 1.2$$, below which there are no samples. What's going on? It turns out this lower bound is $$H(p, p)$$. Empirically, this looks correct:


```python
print('H(p, p) = {:0.3f}'.format(cross_entropy(p, p)))
```

    H(p, p) = 1.191


Now let's prove this. The precise claim is as follows: Suppose $$p$$ and $$q$$ are length-$$n$$ multinomial PDFs with no zero probabilities. Then the function $$H(p, q)$$, with $$p$$ fixed and $$q$$ varying, has a global minimum at $$q = p$$. To show this, we use the fact that any discrete PDF $$q = (q(x_{1}), \ldots, q(x_{n}))$$ with no zero entries can be written using a softmax parametrisation:

\begin{equation}
q(x_{j}) = \frac{e^{t_{j}}}{Z},
\end{equation}

with $$Z$$ defined to be the normalisation constant $$\sum_{i=1}^{n} e^{t_{i}}$$.

### Step 1:
Fix $$p$$ and re-write $$H(p, q)$$ by replacing $$q(x_{i})$$ with $$e^{t_{i}}/Z$$:

\begin{equation}
H(p, q) = -\sum_{i=1}^{n} p(x_{i}) \log(q(x_{i})) = -\sum_{i=1}^{n} p(x_{i}) \log \left( \frac{e^{t_{i}}}{Z} \right) = 
\sum_{i=1}^{n} p(x_{i})(\log(Z) - t_{i}) = \log(Z) - \sum_{i=1}^{n} p(x_{i}) t_{i}.
\end{equation}
### Step 2:
Now we find local optima for $$H(p, q)$$. The first step is to solve $$\partial H/\partial t_{j} = 0$$. To do this, observe that because $$Z = \sum_{i=1}^{n} e^{t_{i}}$$, we have $$\partial Z/\partial t_{j} = e^{t_{j}}$$, so that:

\begin{equation}
\frac{\partial \log(Z)}{t_{j}} = \frac{e^{t_{j}}}{Z} = q(x_{j}),
\end{equation}

and hence:

\begin{equation}
\frac{\partial H(p, q)}{\partial t_{j}} = q(x_{j}) - p(x_{j}).
\end{equation}

This partial derivative is zero exactly when $$q(x_{j}) = p(x_{j})$$.
### Step 3:
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
\frac{Z \cdot \frac{\partial e^{t_{i}}}{\partial t_{j}} - e^{t_{i}} \cdot e^{t_{j}}}{Z^{2}} = 
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
\nabla^{2} H(p, q) = diag(Q) - QQ^{t},
\end{equation}

where $$Q = (q(x_{1}), \ldots, q(x_{n}))$$ is written as an $$n$$-dimensional column vector, and $$diag(Q)$$ is the $$n \times n$$ matrix whose diagonal is defined by $$Q$$, and whose off-diagonal entries are zero. Observe now that for $$v \in \mathbb{R}^{n}$$, we have:

\begin{equation}
v \nabla^{2} H(p, q) v^{t} = v diag(Q) v^{t} - v QQ^{t} v^{t} = \\
\sum_{i=1}^{n} v_{i}^{2} q(x_{i}) - \sum_{i=1}^{n} v_{i}^{2} q(x_{i})^{2} =
\sum_{i=1}^{n} v_{i}^{2} q(x_{i}) (1 - q(x_{i})).
\end{equation}
Here, each summand $$v_{i}^{2} q(x_{i}) (1 - q(x_{i}))$$ is non-negative, being the product of a square, a probability and its complementary probability. It follows that the whole sum is non-negative. Since $$p$$ and $$q$$ have no zero probabilities by assumption, this sum can be zero only if each $$v_{i}$$ is zero, so $$\nabla^{2} H(p, q)$$ is positive definite and the minimum $$q = p$$ is global.


```python

```
