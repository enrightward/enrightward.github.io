---
title: "Convolutions of random variables — Part 1: Introduction"
date: 2021-03-14
categories: [statistics, convolutions]
tags: [statistics, convolutions, uniform distribution, sums of random variables]
preview_image: /enrightward.github.io/assets/img/convolutions/convolution-formula.png
---

![Desktop View](/assets/img/convolutions/convolution-formula.png)

## 1. Introduction

Suppose that $$X$$ and $$Y$$ are two continuous random variables. Given their PDFs PDFs $$f_{X}$$ and $$f_{Y}$$, how do you compute the $$f_{X + Y}$$ of the sum of $$X+Y$$? In this post, we'll define, compute examples of and prove some facts about the _convolution_ $$f_{X} * f_{Y}$$ of $$f_{X}$$ and $$f_{Y}$$, itself a PDF, and moreover equal to $$f_{X + Y}$$, provided $$X$$ and $$Y$$ are independent. A minor adaptations yield a method for computing $$f_{Y - X}$$. We will explicitly compute the PDFs $$f_{X+Y}$$ and $$f_{X-Y}$$ for $$X, Y \sim U(0, 1)$$. Although doing all the algebra rigorously is quite fiddly, it turns out there is a nice way of visualising these convolutions is the area of intersection of two colliding rectanlges in the plane, which I [learnt about on Wikipedia](https://en.wikipedia.org/wiki/Convolution). I have also made my own version of this visualisation below, using the [desmos graphical calculator app](https://www.desmos.com/).

**Note:** This post is quite technical and calculation-heavy. To get a high-level feel for convolutions, together with a discussion of convolutional neural networks, see for instance [this lovely post from Christopher Olah](https://colah.github.io/posts/2014-07-Understanding-Convolutions/).

## 2. Convolutions

As motivation, suppose first $$X$$ and $$Y$$ are independent discrete random variables, taking values on the integers, and let $$Z := X + Y$$. Then the PDF of $$Z$$ is given by:

\begin{align}
\mathbb{Pr}(Z = z) = \sum_{k \in \mathbb{Z}} \mathbb{Pr}(X = k) \, \mathbb{Pr}(Y = z-k).
\end{align}

Now assume $$X$$ and $$Y$$ are continuous, independent random variables with PDFs $$f_{X}$$ and $$f_{Y}$$, taking values on the whole real line. What is the analogue of the above summation formula? We have for any $$z \in \mathbb{R}$$ that:

\begin{align}
F_{Z}(z) = \mathbb{Pr}(Z \le z) = \int \int_{S(z)} f_{X}(x) \, f_{Y}(y) \, dy \, dx,
\end{align}

where $$S(z) := \{ (x, y) \in \mathbb{R}^{2} \mid x + y \le z \}$$. We can integrate over $$S(z)$$ by allowing $$x$$ to vary freely, and $$y$$ to vary over the half-open interval $$(-\infty, z-x]$$:

\begin{align}
F_{Z}(z) = \int_{-\infty}^{\infty} f_{X}(x) \left\\{ \int_{-\infty}^{z-x} f_{Y}(y) \, dy \right\\} dx
= \int_{-\infty}^{\infty} f_{X}(x) \, F_{Y}(z-x) \, dx.
\end{align}

To pass to the PDF $$f_{Z}(z)$$, we must take the derivative of $$F_{Z}(z)$$ with respect to $$z$$. Using the [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule) to interchange the order of differentiation and integration, we have

\begin{align}
f_{Z}(z) &= F_{Z}^{\prime}(z) = 
\frac{d}{dz} \left( \int_{-\infty}^{\infty} f_{X}(x) \, F_{Y}(z-x) \, dx \right) \newline
&= \int_{-\infty}^{\infty} \frac{d}{dz} \left( f_{X}(x) \, F_{Y}(z-x) \, dx \right) \newline
&= \int_{-\infty}^{\infty} f_{X}(x) \, f_{Y}(z-x) \, dx,
\end{align}

where the equality uses the fact that $$f_{X}(x)$$ is independent of $$z$$, and also that the derivative of $$F_{Y}(z-x)$$ with respect to $$z$$ is $$f_{Y}(z-x)$$, by the chain rule. We refer to the integral as the _convolution_ of the functions $$f_{X}$$ and $$f_{Y}$$. In this language, we have shown that the PDF of the sum of two independent random variables is given by the convolution of their PDFs.

## 3. PDFs of sums and differences of two random variables

We showed in the previous section that

\begin{align}
f_{X+Y}(z) = \int_{-\infty}^{\infty} f_{X}(x) \, f_{Y}(z-x) \, dx, \newline
F_{X+Y}(z) = \int_{-\infty}^{\infty} f_{X}(x) \, F_{Y}(z-x) \, dx.
\end{align}

A similar argument, starting with the variable $$Z = Y - X$$, shows that 

\begin{align}
f_{Y-X}(z) &= \int_{-\infty}^{\infty} f_{X}(x) \, f_{Y}(z+x) \, dx, \newline
F_{X-Y}(z) &= \int_{-\infty}^{\infty} f_{X}(x) \, F_{Y}(z+x) \, dx.
\end{align}

In the next section, we compute examples these PDFs in the case where $$X, Y \sim U(0, 1)$$.

## 4. The sum of two uniform random variables

Suppose now that $$X, Y \sim U(0, 1)$$. To calculate the sum and difference of their PDFs, we introduce "indicator function" notation: For any element $$x \in \mathbb{R}$$ and subset $$S \subseteq \mathbb{R}$$, define

\begin{align}
{\mathbb{1}}\_{S}(x) := \left\\{ \begin{array}{ll}
1, & \textrm{if } x \in S,  \newline
0, & \textrm{otherwise.}
\end{array} \right.
\end{align}

Then for the uniform random variables $$X$$ and $$Y$$,

\begin{align}
f_{X}(x) &= f_{Y}(x) = {\mathbb{1}}\_{\lbrack 0, 1 \rbrack}(x), \newline
F_{X}(x) &= F_{Y}(x) = x \cdot {\mathbb{1}}\_{\lbrack 0, 1 \rbrack}(x).
\end{align}

We now compute $$f_{X+Y}(z)$$. By the formula above, $$f_{X}(x) = \mathbb{1}_{[0, 1]}(x)$$, but what is $$f_{Y}(z -x)$$? To answer this, note that $$0 \le z - x \le 1$$ if and only if $$z - 1 \le x \le z$$, and so:

\begin{align}
f_{Y}(z - x) = {\mathbb{1}}\_{[0, 1]}(z - x) = {\mathbb{1}}\_{[z-1, z]}(x).
\end{align}

Using the convolution formula, we now have:

\begin{align}
f_{X+Y}(z) &= \int_{-\infty}^{\infty} f_{X}(x) \, f_{Y}(z-x) \, dx \newline
&= \int_{-\infty}^{\infty} {\mathbb{1}}\_{[0, 1]}(x) \, {\mathbb{1}}\_{[z-1, z]}(x) \, dx \newline
&= \int_{-\infty}^{\infty} {\mathbb{1}}\_{[0, 1] \cap [z-1, z]}(x) \, dx.
\end{align}

The last equality holds because the product $$\mathbb{1}_{S}(x) \cdot \mathbb{1}_{T}(x)$$ equals $$1$$ if and only if $$x$$ belongs to both $$S$$ and $$T$$, and is zero otherwise, showing the general identity $$\mathbb{1}_{S} \cdot \mathbb{1}_{T} = \mathbb{1}_{S \cap T}$$. Set now $$I: = [0, 1]$$ and $$J := [z-1, z]$$ to ease notation. Clearly, $$I \cap J$$ is an interval whose endpoints, hence length, depend on $$z$$. Concretely:

\begin{align}
I \cap J = \left\\{ \begin{array}{ll}
\emptyset, & \textrm{if } z \le 0, \; \textrm{or} \; z \ge 2, \newline
[z-1, 1], & \textrm{if } 1 \le z \le 2, \newline
[0, z], & \textrm{if } 0 \le z \le 1.
\end{array} \right.
\end{align}

Geometrically, the inequality conditions correspond to the fact that $$I$$ and $$J$$ either (i) Don't overlap at all, or else they overlap partially, and (ii) Either and $$I$$ is on the left, or else (iii) $$J$$ is on the left. For an interval $$S = (a, b)$$, the integral over the whole real line of $$\mathbb{1}_{S}(x)$$ is equal to the length $$b-a$$ of $$S$$. It follows that 

\begin{align}
f_{X+Y}(z) &= \int_{-\infty}^{\infty} {\mathbb{1}}\_{I \cap J}(x) \, dx =
\left\\{ \begin{array}{ll}
0, & \textrm{if } z \le 0, \; \textrm{or} \; z \ge 2, \newline
2-z, & \textrm{if } 1 \le z \le 2, \newline
z, & \textrm{if } 0 \le z \le 1.
\end{array} \right.
\end{align}

We plot this, using the `matplotlib` code in the cell below. The function `graph_piecemeal` is (only slightly) adapted from code posted by stackoverflow user [rputikar](https://stackoverflow.com/users/1888184/rputikar), in response to [this question](https://stackoverflow.com/questions/14000595/graphing-an-equation-with-matplotlib).


```python
import numpy as np  
import matplotlib.pyplot as plt

def _set_ax_colour(ax, colour):
    """Set the spine, label and ticks of the 
    axis colour to `colour`"""
    ax.spines['bottom'].set_color(colour)
    ax.spines['top'].set_color(colour)
    ax.spines['left'].set_color(colour)
    ax.spines['right'].set_color(colour)
    ax.xaxis.label.set_color(colour)
    ax.yaxis.label.set_color(colour)
    ax.title.set_color(colour)
    ax.tick_params(axis='x', colors=colour)
    ax.tick_params(axis='y', colors=colour)
    
def graph_piecemeal(boundaries, formulae, resolution, figsize, title):
    """Display a piecemeal graph defined by the list of 
    strings `formulae`, separated by the list of numbers 
    `boundaries`. Each piece of the x_axis is broken into 
    `resolution` parts for graphing. There must be one less 
    formula than boundary."""
    assert len(formulae) == len(boundaries) - 1
    
    # setup axes for dark background
    fig, ax = plt.subplots(figsize=figsize)
    _set_ax_colour(ax, 'white')
    ax.set_title(title, fontsize=12)
    
    # define `intervals` as consecutive boundary points.
    intervals = [np.linspace(a, b, resolution) for a, b in zip(boundaries, boundaries[1:])]
    assert len(intervals) == len(formulae)
    
    # graph pieces (= (interval, formula) pairs) in a loop
    for interval, formula in zip(intervals, formulae):
        x = np.array(interval)
        y = eval(formula)
        plt.plot(x, y, color='blue')
        
    plt.show()
```


```python
# Define the boundaries between the formulae. They correspond to the four regions
# z < 0, 0 < z < 1, 1 < z < 2, z > 2 in the definition of f_{X+Y}
boundaries = [-1, 0, 1, 2, 3]

# Define the different formulae for each region
formulae = ['x*0', 'x', '2 - x', 'x*0']

resolution = 5
figsize = (10, 6)
title = 'PDF of X+Y, where X and Y uniform'
graph_piecemeal(boundaries, formulae, resolution, figsize, title)
```

![png](/assets/img/convolutions/indicator_plus.png)

##  5. The difference of two uniform random variables

We now compute the PDF of $$Z = Y - X$$. The method parallels that of the sum $$X+Y$$, so we skip some steps. As noted above, the general formula is:

\begin{align}
f_{Y-X}(z) &= \int_{-\infty}^{\infty} f_{X}(x) \, f_{Y}(z+x) \, dx.
\end{align}

This time, to find $$f_{Y}(z+x)$$ we note that $$0 \le z + x \le 1$$ if and only if $$-z \le x \le 1 - z$$, and so:

\begin{align}
f_{Y}(z + x) = \mathbb{1}_{[0, 1]}(z + x) = \mathbb{1}_{[-z, 1-z]}(x).
\end{align}

The convolution now reads:

\begin{align}
f_{X+Y}(z) = \int_{-\infty}^{\infty} \mathbb{1}_{[0, 1] \cap [-z, 1-z]}(x) \, dx.
\end{align}

If now $$I: = [0, 1]$$ and $$J := [-z, 1-z]$$, then  $$I \cap J$$ depends on $$z$$ as follows:

\begin{align}
I \cap J = \left\\{ \begin{array}{ll}
\emptyset, & \textrm{if } z \le -1, \; \textrm{or} \; z \ge 1, \newline
[-z, 1], & \textrm{if } -1 \le z \le 0, \newline
[0, 1-z], & \textrm{if } 0 \le z \le 1.
\end{array} \right.
\end{align}

It follows that:

\begin{align}
f_{Y-X}(z) &= \int_{-\infty}^{\infty} \mathbb{1}_{I \cap J}(x) \, dx =
\left\\{ \begin{array}{ll}
0, & \textrm{if } z \le -1, \; \textrm{or} \; z \ge 1, \newline
1 + z, & \textrm{if } -1 \le z \le 0, \newline
1 - z, & \textrm{if }  0 \le z \le 1.
\end{array} \right.
\end{align}

As above, let's display the graph of $$f_{Y-X}(z)$$.


```python
# Define the boundaries between the formulae. They correspond to the four regions
# z < 0, 0 < z < 1, 1 < z < 2, z > 2 in the definition of f_{X+Y}
boundaries = [-2, -1, 0, 1, 2]

# Define the different formulae for each region
formulae = ['x*0', '1+x', '1-x', 'x*0']

resolution = 5
figsize = (10, 6)
title = 'PDF of Y-X, where X and Y uniform'
graph_piecemeal(boundaries, formulae, resolution, figsize, title)
```

![png](/assets/img/convolutions/indicator_minus.png)

## 6. Intuition about convolutions of uniform PDFs

So $$f_{Y-X}(x)$$ is the same shape as $$f_{X+Y}(x)$$, but shifted one unit to the left. Could we have seen this in advance, you ask? Why, yes! It follows from the fact that if $$X \sim U(0, 1)$$, then $$X-1 = -X$$ as random variables, since:

\begin{align}
f_{-X}(x) = \mathbb{1}_{[-1, 0]}(x) = f_{X-1}(x), 
\end{align}

implying also $$Y-X=X+Y-1$$. It follows:

\begin{align}
f_{Y-X}(x) = f_{X+Y-1}(x) = f_{X+Y}(x+1),
\end{align}

where the last equality holds because $$f_{Z-a}(z) = f_{Z}(z+a)$$ for any random variable $$Z$$ and constant $$a$$. Note that the equivalence of $$-X$$ and $$X-1$$ is also quite intuitive, for it says that if you sample a real number $$x$$ uniformly from $$[0, 1]$$, then either subtracting $$1$$ to get $$x-1$$, or negating to get $$-x$$ are probabilistically the same: You land (uniformly) in the range $$[-1, 0]$$.

You might also ask: Could have predicted the "pyramid" shape of $$f_{X+Y}(z)$$ in the first place? The answer is also yes, I think, at least once you know the formula:

\begin{align}
f_{X+Y}(z) = \int_{-\infty}^{\infty} \mathbb{1}_{[0, 1] \cap [-z, 1-z]}(x) \, dx.
\end{align}

This integral invites us to consider the indicator function of the set $$I \cap J$$, where $$I := [0, 1]$$ and $$J := [-z, 1-z]$$. This function is constant and equal to one on $$I \cap J$$, and is equal to zero every where else, so the value of this integral at any given $$z$$ is the length of $$I \cap J$$, or equivalenty, the area under the curve $$y=1$$ on $$I \cap J$$. The first interval, $$I$$, is fixed, and both have constant length $$1$$, but $$J$$ slides around as we vary $$z$$. So, computing the length of $$I \cap J$$ for any particular $$z$$ is the same as understanding how much $$I$$ and $$J$$ overlap for $$z$$. The rate of change of $$f_{X+Y}(z)$$, as $$z$$ changes, is simply the rate of change of this overlap area. Since the overlapping areas are squares, overlap increases linearly until it is complete (corresponding to the apex of the pyramid), then decreases linearly at the same rate. This is shown below in the following animation, made in [desmos](https://www.desmos.com/calculator/): 

![gif](/assets/img/convolutions/indicator.gif)

## 7. Roundup 

We defined, for two continuous random variables $$X$$ and $$Y$$, the _convolution_ $$f_{X} * f_{Y}$$ of $$f_{X}$$ and $$f_{Y}$$. The convolution is itself a PDF. In the case where $$X$$ and $$Y$$ are independent, $$f_{X} * f_{Y}$$ is equal to the PDF $$f_{X + Y}$$ of the sum of $$X+Y$$. We then computed the PDFs for $$f_{X+Y}$$ and $$f_{X-Y}$$ explicitly. Algebraiclly, the answer in each case was an ugly-looking four-part piecemeal function. Visually, however, these PDFs are two, translated copies of the same "pyramid" function. Using some basic reasoning and a visualisation of the changing area of colliding rectangles in the plane, we were saw where the pyramid came from. Starting from the observation that if $$X \sim U(0,1)$$, then $$1-X=-X$$, we were able to explain the translation. In the [next post](https://enrightward.github.io/enrightward.github.io/posts/convolutions-part-2/), we'll use convolutions to explicitly compute and visualise the PDF $$f_{aX+bY}$$ of a linear combination of $$X \sim U(0,1)$$.
