---
title: "Convolutions of RVs — Part 2: A linear combination of uniform RVs"
date: 2021-03-16
categories: [statistics, convolutions]
tags: [statistics, convolutions, uniform distribution, linear combinations random variables]
preview_image: /enrightward.github.io/assets/img/convolutions/mesa.jpeg
---

![Desktop View](/assets/img/convolutions/mesa.jpeg)

## 1. Introduction

In the [last post](https://enrightward.github.io/enrightward.github.io/posts/convolutions-part-1/), we defined the _convolution_ of the PDFs $$f_{X}$$ and $$f_{Y}$$ of two independent, continuous random variables $$X$$ and $$Y$$, and showed that it computed the PDF $$f_{X + Y}$$ of the sum of $$X+Y$$. Minor adaptations also yielded a method for computing $$f_{Y - X}$$. We visualised both these PDFs in the case $$X, Y \sim U(0, 1)$$, and saw that they differed only by a translation along the $$x$$-axis, both having the same pyramid shape. We explained this resemblance, starting from the observation that if $$X \sim U(0,1)$$, then $$1-X=-X$$. Finally, we were able to visualise the convolution $$f_{X+Y}$$ as the intersection area of two colliding rectanlges in the plane, explaining the pyramid shape.

In this post, we consider the formula for linear combinations $$f_{aX + bY}$$. The resulting shape will be a "pyramid with plateau", whose flat portion is of length $$\lvert b - a \rvert$$.

## 2. Affine transformations of random variables

We want to compute the PDF of $$Z := aX + bY$$, where $$a, b \in \mathbb{R}$$ are arbitrary. To do this, we derive some simple rules describing how a PDF changes as the underyling random variable is scaled and translated. If $$a$$ is positive, then the conditions $$aX \le x$$ and $$X \le x/a$$ are equivalent, so

\begin{align}
F_{aX}(x) &= \mathbb{Pr}(aX \le x) = \mathbb{Pr} \left( X \le \frac{x}{a} \right) = 
F_{X} \left( \frac{x}{a} \right), \quad \textrm{implying} \newline
f_{aX}(x) &= \frac{d}{dx} F_{X} \left( \frac{x}{a} \right) = \frac{1}{a} f_{X} \left( \frac{x}{a} \right).
\end{align}

If $$a$$ is negative, then $$aX \le x$$ is the same as $$X \ge x/a$$, so 

\begin{align}
F_{aX}(x) &= \mathbb{Pr}(aX \le x) = \mathbb{Pr}\left( X \ge \frac{x}{a} \right) = 
1 - \mathbb{Pr}\left( X \le \frac{x}{a} \right) = 
1 - F_{X} \left( \frac{x}{a} \right), \quad \textrm{implying} \newline
f_{aX}(x) &= \frac{d}{dx} F_{X} \left( \frac{x}{a} \right) = -\frac{1}{a} f_{X} \left( \frac{x}{a} \right).
\end{align}

We can group the two cases together:

\begin{align}
f_{aX}(x) &= \left\vert \frac{1}{a} \right\vert f_{X} \left( \frac{x}{a} \right).
\end{align}

## 3. Simplifying the integrand of the convolution

We'll use the identities from the previous section to simplify the integrand of the convolution $$f_{aX + bY}$$, in the case where $$X, Y \sim U(0, 1)$$. By the definition of convolution, we have:

\begin{align}
f_{aX+bY}(z) &= \int_{-\infty}^{\infty} f_{aX}(x) \, f_{bY}(z-x) \, dx.
\end{align}

Inside the integrand, $$z$$ functions as a constant. Since $$X, Y \sim U(0, 1)$$, the factors $$f_{aX}$$ and $$f_{bY}$$ in the integrand are indicator functions. Let's apply the rules from the previous section to rewrite them. The first factor looks like:

\begin{align}
f_{aX}(x) &= \left\vert \frac{1}{a} \right\vert f \left( \frac{x}{a} \right) = 
\left\vert \frac{1}{a} \right\vert {\mathbb{1}}\_{[0, 1]} \left( \frac{x}{a} \right) = 
\left\\{ \begin{array}{ll}
\left\vert \frac{1}{a} \right\vert {\mathbb{1}}\_{[0, a]}(x), & \textrm{if } a > 0, \newline
\left\vert \frac{1}{a} \right\vert {\mathbb{1}}\_{[a, 0]}(x), & \textrm{if } a < 0.
\end{array} \right.
\end{align}

The last equality follows because $$0  \le \frac{x}{a} \le 1$$ is equivalent either to $$0  \le x \le a$$ or $$a  \le x \le 0$$, depending on the sign of $$a$$. The second factor is:

\begin{align}
f_{bY}(z-x) &= \left\vert \frac{1}{b} \right\vert f_{Y} \left( \frac{z-x}{b} \right) =  
\left\vert \frac{1}{b} \right\vert {\mathbb{1}}\_{[0, 1]} \left( \frac{z-x}{b} \right) =
\left\\{ \begin{array}{ll}
\left\vert \frac{1}{b} \right\vert {\mathbb{1}}\_{[z-b, z]}(x), & \textrm{if } b > 0, \newline
\left\vert \frac{1}{b} \right\vert {\mathbb{1}}\_{[z, z-b]}(x), & \textrm{if } b < 0.
\end{array} \right.
\end{align}

The last equality follows because $$0  \le \frac{z-x}{b} \le 1$$ is equivalent either to $$z-b \le x \le z$$ or $$z \le x \le z-b$$, depending on the sign of $$b$$. Our calculations say the integrand takes slightly different forms in each of the four cases $$a, b \in \{ +1, -1 \}$$, but the difference affects only the order in which the endpoints of the intervals in the indicator functions are written. To accommodate this, we write the integrand in a unified form, using some non-standard notation:

\begin{align}
f_{aX+bY}(z) = 
\left\vert \frac{1}{ab} \right\vert \int_{-\infty}^{\infty} {\mathbb{1}}\_{I}(x) \, {\mathbb{1}}\_{J}(x) \, dx = 
\left\vert \frac{1}{ab} \right\vert \int_{-\infty}^{\infty} {\mathbb{1}}\_{I \cap J}(x) \, dx,
\end{align}

where:

\begin{align}
I &:= \textrm{sgn}(a)[0, a] = 
\left\\{ \begin{array}{ll}
\lbrack 0, a \rbrack, & \textrm{if } a > 0, \newline
\lbrack a, 0 \rbrack, & \textrm{if } a < 0,
\end{array} \right.
\end{align}

and the meaning of $$J := \textrm{sgn}(b)[z-b, z]$$ is similar.

## 4. Computing the convolution integral

To actually compute this integral, of course, we still need to distinguish the four cases $$a, b \in \{ +1, -1 \}$$. I will do this here only for the case where $$a$$ and $$b$$ are both positive. We set 
$$I := [0, a]$$ and $$J := [z-b, z]$$, note that $$I$$ and $$J$$ have lengths $$a$$ and $$b$$, respectively,
and consider how $$I \cap J$$ depends on $$z$$. Technically, there are two more subcases: Either $$0 < a < b$$, or $$0 < b < a$$. However, the setup is symmetric in $$a$$ and $$b$$: Since $$X$$ and $$Y$$ are both $$\sim U(0, 1)$$, it follows $$aX + bY = aY + bX$$ as random variables. This means we can compute the convolution integral in either case and simply swap $$a$$ and $$b$$ to obtain the formula in the other case. So, we suppose $$0 < a < b$$, meaning that $$J$$ can contain $$I$$ for certain values of $$z$$. Then 

\begin{align}
I \cap J = \left\\{ \begin{array}{ll}
\emptyset, & \textrm{if } z \le 0, \; \textrm{or} \; z \ge a+b, \newline
[z-b, a], & \textrm{if } b \le z \le a+b, \newline
[0, a], & \textrm{if } a \le z \le b, \newline
[0, z], & \textrm{if } 0 \le z \le a.
\end{array} \right.
\end{align}

Geometrically, the first condition means that the intervals $$I$$ and $$J$$ do not overlap. In the other three cases, they do. In the second case, $$I$$ is left of $$J$$ and partially overlaps; in the third, $$I$$ is contained in $$J$$, so the overlap is all of $$I$$; in the last case, $$I$$ is right of $$J$$ with partial overlap. The piecemeal formula for $$I \cap J$$ implies:

\begin{align}
f_{aX+bY}(z) &= 
\left \lvert \frac{1}{ab} \right \rvert 
\int_{-\infty}^{\infty} \mathbb{1}_{I \cap J}(x) \, dx =
\left\\{ \begin{array}{ll}
0, & \textrm{if } z \le 0, \; \textrm{or} \; z \ge a+b, \newline
\frac{a+b-z}{\lvert ab \rvert}, & \textrm{if } b \le z \le a+b, \newline
\frac{1}{\lvert b \rvert}, & \textrm{if } a \le z \le b, \newline
\frac{z}{\lvert ab \rvert}, & \textrm{if } 0 \le z \le a.
\end{array} \right.
\end{align}

We graph this below using the code in the next cell, familiar from last time.


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
# Define the boundaries between the formulae. They correspond to the five regions
# z < 0, 0 < z < a, a < z < b, b < z < a + b, z > a+b in the definition f_{aX+bY}
a = 2
b = 10
d = (b-a)*0.5
boundaries = [-d, 0, a, b, a+b, a+b+d]

# Define the different formulae for each region
#formulae = ['x*0', 'x', 'x-x+a', 'a+b-x', 'x*0']
formulae = ['x*0', 'x/abs(a*b)', '(x-x+1)/(abs(b))', '(a+b-x)/abs(a*b)', 'x*0']
resolution = 5
figsize = (10, 6)
title = 'PDF of aX+bY, where X and Y uniform'
graph_piecemeal(boundaries, formulae, resolution, figsize, title)
```

![png](/assets/img/convolutions/indicator_ab.png)

This time, the PDF has the shape of a flattened pyramid. The plateau on top ranges between $$a=2$$ and $$b=10$$ (and since the roles of $$X$$ and $$Y$$ are symmetric in this formula, it would range between $$b$$ and $$a$$ if $$b$$ were the smaller constant).

## 5. Visualising this integral

As in the [previous post](https://enrightward.github.io/enrightward.github.io/posts/convolutions-part-1/), we can both explain this shape in words, and visualise it (below, in the next cell) with the [desmos calculator](https://www.desmos.com/). Verbal explanation: The integral

\begin{align}
f_{aX+bY}(z) &= 
\left \lvert \frac{1}{ab} \right \rvert 
\int_{-\infty}^{\infty} \mathbb{1}_{I \cap J}(x) \, dx
\end{align}

where $$I := [0, a]$$ and $$J := [z-b, z]$$, invites us to consider the snapshot at time $$t=z$$ as two rectangles of (in general) differing height and width collide, then pass through each other. Indeed, the integrand $$\mathbb{1}_{I \cap J}(x)$$ computes the area of this intersection as a function of $$t=z$$ (note that the position of $$J$$ depends on $$z$$). As $$J$$ approaches $$I$$ from the left, there is at first no overlap, then collision, then a steadily, linearly increasing overlap until the wider of $$I$$ and $$J$$ contains the other, then a plateau as the narrower of the two passes inside the wider, then a symmetrical process of linearly decreasing intersection until $$J$$ leaves $$I$$ forever.

![gif](/assets/img/convolutions/ab-indicator.gif)

## 6. Roundup

We computed the PDF of an arbitrary linear combination $$Z := aX + bY$$ of uniform random variables $$X$$ and $$Y$$. Algebraically, it is an unpleasant-looking piecemeal function, but visually, it's just a flattened pyramid. We explained this shape with a story about the intersection area of non-identical colliding rectangles, and visualised this story using a [desmos graphical calculator](https://www.desmos.com/).
