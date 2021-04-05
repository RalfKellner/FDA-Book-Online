#!/usr/bin/env python
# coding: utf-8

# # Math - General Concepts
# In the beginning, there is set theory. A **set** is a collection of distinct elements, e.g., $\lbrace 1, 2, 3 \rbrace$. Important sets for us are numerical sets, i.e., the set of natural numbers $\mathbb{N}$ and the set of real numbers $\mathbb{R}$. When we declare a single number, a **scalar**, from a set, we write $n \in \mathbb{N}$ or $r \in \mathbb{R}$. A scalar can be a number which is used to describe an attribute of a person or an object, e.g., the size, the price, ... when we are in need to record multiple attributes, we use the Cartesian product of a set. For instance the Cartesian product of $\mathbb{R}$ two times is $\mathbb{R}^{2}$ and we write $\boldsymbol{x} = (x_1, x_2) \in \mathbb{R}^{2}$ to highlight that $\boldsymbol{x}$ is a member of $\mathbb{R}^{2}$; $\boldsymbol{x}$ could be the information of a person where $x_1$ represents body height and $x_2$ age of the person. Thus, you should remember that the position in an element of a Cartesian product is informative and can not be permuted arbitrarily. Generally speaking, elements of Cartesian products are arrays of numbers and when we use them for mathematical operations, we call them **vectors**. A vector can be considered as a (one-dimensional) special case of a **matrix** which is a two dimensional array of numbers. Assume, you want to keep track of body height and age of multiple persons, you need a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ where each row $i = 1, ..., m$ represents a different person and the columns $j = 1, ..., n$ contain attributes of persons. For instance, if we collect information of three persons, the matrix containing this information would look like:
# 
# $$\boldsymbol{A} = \begin{pmatrix}
# a_{1,1} & a_{1,2}  \\ 
# a_{2,1} & a_{2,2}  \\
# a_{3,1} & a_{3,2}  \\ 
# \end{pmatrix}$$
# 
# Sometimes, we want to process data according to functional relationships, e.g., we may want to derive the body mass index with the information of age, body height and weight. Mathematically, we use a **function**, or more generally speaking, a **mapping**. This means we map elements from one set to elements of another set according to a certain rule. This is denoted as 
# 
# $$f: M \to N $$
# 
# and tells us that according to rule $f$ we map an element of $M$ to an element of $N$. Technically, it must hold that only a unique element of $N$ corresponds to each element of $M$. This means whenever we choose the same element of $M$ under a specific rule $f$ we get the same element of $N$ over and over again. Or more easily spoken,  whenever we use the same argument for a function, it must lead to the same result. A mapping is a very general concept. It includes one dimensional functions $f: \mathbb{R} \to \mathbb{R}$ and multidimensional functions $f: \mathbb{R}^n \to \mathbb{R}$ or other mathematical forms of mappings that we do not need in this course.
# 
# # Linear Algebra
# 
# 
# ## Matrices and Vectors
# 
# One of the most important things for data science is to conduct basic mathematical operations with matrices and to know about typical concepts of matrices and their decompositions. Popular matrices are:
# 
# * square matrix: $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, which means the number of rows and columns are identical
# * transposed matrix: $\boldsymbol{A}^T = a_{ji}, i = 1, ..., m \text{ and } j = 1, ..., n$, which is the matrix we get if we switch rows and columns
# * diagonal matrix: a square matrix $\boldsymbol{D} \in \mathbb{R}^{n \times n}$ whose elements are all equal to zero except those on its diagonal 
# 
# $$\boldsymbol{D} = 
# \begin{pmatrix} d_1 & 0 & \cdots & 0\\
#     0 & d_2 & \cdots & 0\\
#     \vdots & \vdots & & \vdots\\
#     0 & 0 & \cdots & d_n
# \end{pmatrix} = \text{diag}\lbrace d_i \rbrace$$
# 
# * identiy matrix: a square matrix $\boldsymbol{I} \in \mathbb{R}^{n \times n}$ whose elements are all equal to zero except those on its diagonal which are all equal to one
# 
# $$\boldsymbol{I} =
# \begin{pmatrix}
#     1 & 0 & \cdots & 0\\
#     0 & 1 & \cdots & 0\\
#     \vdots & \vdots & 1 & 0\\
#     0 & 0 & \cdots & 1
# \end{pmatrix} $$
# 
# * inverse matrix: The inverse of a matrix $\boldsymbol{A}^{-1}$ is defined as $\boldsymbol{A} \boldsymbol{A}^{-1} = \boldsymbol{A}^{-1} \boldsymbol{A} = \boldsymbol{I}$, however, not for every matrix an inverse exists, but if it exists it is unique; a matrix with an inverse is called **regular**, while it is called **singular** if no inverse exists
# 
# Basic matrix operations are, scalar multiplication, addition and matrix multiplication. The first two are quite intuitive. Given a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ and a scalar $\lambda \in \mathbb{R}$, scalar multiplication is defined by:
# 
# $$\lambda \cdot \boldsymbol{A} = \lambda \cdot 
# \begin{pmatrix}
#     a_{11} & \cdots & a_{1n}\\
#     a_{21} & \cdots & a_{2n}\\
#     \vdots &        & \vdots\\
#     a_{m1} & \cdots & a_{mn}
#     \end{pmatrix} =  
#     \begin{pmatrix}
#     \lambda \cdot a_{11} & \cdots & \lambda \cdot a_{1n}\\
#     \lambda \cdot a_{21} & \cdots & \lambda \cdot a_{2n}\\
#     \vdots &        & \vdots\\
#     \lambda \cdot a_{m1} & \cdots & \lambda \cdot a_{mn}
# \end{pmatrix}$$
# 
# Matrix addition of two matrices $\boldsymbol{A},\boldsymbol{B} \in \mathbb{R}^{m \times n}$ is conducted by elementwise addition of each element:
# 
# $$A + B = 
# \begin{pmatrix}
#     a_{11} & \cdots & a_{1n}\\
#     a_{21} & \cdots & a_{2n}\\
#     \vdots &        & \vdots\\
#     a_{m1} & \cdots & a_{mn}
# \end{pmatrix} + 
# \begin{pmatrix}
#     b_{11} & \cdots & b_{1n}\\
#     b_{21} & \cdots & b_{2n}\\
#     \vdots &        & \vdots\\
#     b_{m1} & \cdots & b_{mn}
# \end{pmatrix} = 
# \begin{pmatrix}
#     a_{11} + b_{11} & \cdots & a_{1n} + b_{1n}\\
#     a_{21} + b_{21} & \cdots & a_{2n} + b_{2n}\\
#     \vdots &        & \vdots\\
#     a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn}
# \end{pmatrix} 
# $$
# 
# Matrix multiplication feels unusal at the start, because it is not conducted elementwise. Given two matrices $\boldsymbol{A} \in \mathbb{R}^{m \times n},\boldsymbol{B} \in \mathbb{R}^{n \times p}$, their product is defined as $\boldsymbol{A} \boldsymbol{B} = \boldsymbol{C} \in \mathbb{R}^{m \times p}$ with: 
# 
# $$c_{ij} = \sum_{k = 1}^{n} a_{ik} b_{kj}$$
# 
# so each element $c_{ij}$ is computed as the dot product of the i-th row from matrix $\boldsymbol{A}$ with the j-th column from matrix $\boldsymbol{B}$. Important to notice is the dimensionality of matrices and its importance for matrix multiplication. Two matrices can only be multiplied if the number of columns from the first matrix is equal to the number of rows from the second matrix. Vectors may be considered as (one dimensional) special cases of matrices consisting of a single row or column. The dot product between two vectors $\boldsymbol{a}, \boldsymbol{b} \in \mathbb{R}^n$ is given by:
# 
# $$\boldsymbol{a}^T\boldsymbol{b}=
# \begin{pmatrix} a_1,a_2,\dots,a_n \end{pmatrix}
# \begin{pmatrix}
# b_1\\b_2\\ \vdots\\ b_n
# \end{pmatrix} = a_1 b_1 + ... + a_n b_n 
# =\sum\limits_{i=1}^n a_i b_i $$
# 
# The square root of the dot product of a vector with its transposed form is called its **norm** and measures its size. Formally, we write:
# 
# $$\Vert\boldsymbol{a}\Vert_2=\sqrt{\boldsymbol{a^T\ a}}=\sqrt{\sum\limits_{i=1}^n a_i^2}$$
# 
# The $2$ in subscript emphasizes that this is the $L^2$ norm of the vector. In general the $L^p$ norm of a vector is defined as:
# 
# $$\Vert\boldsymbol{a}\Vert_p = \left( \sum_i |a_i|^p \right)^{\frac{1}{p}} $$ 
# 
# Besides the $L^2$ norm, only the $L^1 = \sum_i |a_i|$ is commonly used for methods of data analysis. While norms are measuring the size or length of a vector, we also want to take a look at two metrics which can be used for measuring the similarity between two vectors: (1) **euclidean distance** and (2) **cosine similarity**. Euclidean distance $d(\boldsymbol{a}, \boldsymbol{b})$ between two vectors $\boldsymbol{a}, \boldsymbol{b} \in \mathbb{R}^n$ is calculated as: 
# 
# $$d(\boldsymbol{a}, \boldsymbol{b}) = \Vert \boldsymbol{a} - \boldsymbol{b} \Vert_2 = 	\sqrt{\sum_{i = 1}^{n} (a_i - b_i)^2}$$
# 
# Cosine similarity measures similarity by the angle between two vectors and is derived by:
# 
# $$\cos \theta = \frac{\boldsymbol{a}^T \boldsymbol{b} }{ \Vert \boldsymbol{a} \Vert \Vert \boldsymbol{b} \Vert}$$
# 
# The lower the distance or cosine similarity, respectively, the more similar the vectors are. Imagine, you collect financial indicators from companies and each vectors represents one company. With euclidean distance or cosine similarity you are able to analyze how similar companies are.
# 
# 
# ## Linear Equation Systems and Span 
# 
# Solving **linear equation systems** is a necessary expertise for many important tasks related to matrix operations, e.g., matrix inversion. Given $m$ equations with $n$ unknown variables $x_1, ..., x_n$, a system of linear equations is given by: 
# 
# $$ \begin{array}{rrr} 
#         a_{11} x_1 + a_{12} x_2 + ... +  a_{1n} x_n & = & b_1 \\
# 		a_{21} x_1 + a_{22} x_2 + ... +  a_{2n} x_n & = & b_2 \\
# 		\vdots & \vdots & \vdots \\
# 		a_{m1} x_1 + a_{m2} x_2 + ... +  a_{mn} x_n & = & b_m \\ 
#    \end{array}
# $$
# 
# or in matrix notation:
# 
# $$
# \boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}
# $$
# 
# with $\boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{x} \in \mathbb{R}^n, \boldsymbol{b} \in \mathbb{R}^m$. In general, we face three possibilities: (1) the system can not be solved, (2) the system can be solved with an unique solution or (3) the system can be solved with an infinite amount of solutions. We can use the Gauss algorithm or the Gauss-Jordan algorithm, respectively, to find out in which situation we are. To keep this section slim, we omit the illustration of these algorithms, but they can be found in multiple text books or under this [link](https://mathworld.wolfram.com/GaussianElimination.html). Furthermore, another method of solving linear equation systems is given by Cramer'rule which is based on the **determinant** of the matrix $\boldsymbol{A}$. The determinant of a matrix is a unique number which characterizes each (square) matrix. Formally, it is a mapping $\mathbb{R}^{n \times n} \to \mathbb{R}$ with $\text{det}: \boldsymbol{A} \to \text{det}\left( \boldsymbol{A} \right)$. If the determinant is different from zero, a unique solution for the corresponding linear equation system exists, if it is equal to zero, we may face no or infinite solutions.
# 
# One important task for which we need to know how to solve linear equation systems is to analyze how **vector spaces** are generated. Vectors spaces are sets with certain rules (see this [link](https://mathworld.wolfram.com/VectorSpace.html)). Examples are the set of $n$ dimensional vectors $\mathbb{R}^n$ or matrices in $\mathbb{R}^{m \times n}$. Each vector space is generated by a set of basis vectors. To better understand this, we need to define a **linear combination of vectors** $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_k} \in \mathbb{R}^m$ with weights $x_1, x_2, ..., x_k \in \mathbb{R}$ which generates the vector $\boldsymbol{v}$ by:
# 
# $$\boldsymbol{v} = x_1 \boldsymbol{a_1} + x_2 \boldsymbol{a_2} + ... + x_k \boldsymbol{a_k} = \sum_{i = 1}^{k} x_i \boldsymbol{a_i}$$
# 
# The vectors $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_k} \in \mathbb{R}^m$ are **independent**, if the null vector (the vector whose elements are all equal to zero) can only be generated by these vectors when setting all weights $\boldsymbol{x}$ to zero. This is called the **trivial solution** and means the linear equation system:
# 
# $$
# \begin{pmatrix}
#     a_{11} & \cdots & a_{1n}\\
#     a_{21} & \cdots & a_{2n}\\
#     \vdots &        & \vdots\\
#     a_{m1} & \cdots & a_{mn}
# \end{pmatrix}
# \begin{pmatrix}
#     x_1 \\
#     x_2 \\
#     \vdots \\
#     x_n \\
# \end{pmatrix} = 
# \begin{pmatrix}
#     0 \\
#     0 \\
#     \vdots \\
#     0 \\
# \end{pmatrix}
# $$
# 
# has only one solution ($\boldsymbol{x} = \boldsymbol{0}$). If more then one solution exits, the vectors are **dependent**. As a consequence at least one of the vectors $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_k} \in \mathbb{R}^m$ can be generated by the remaining ones. 
# 
# The set of all possible linear vector combinations is called the **span** and denoted as $\langle \boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_k} \rangle$. An important question is, how vector spaces, e.g., $\mathbb{R}^n$ are generated and how many vectors are needed for this task. The short answer is that each vector space can be generated by a number of independent vectors which are called **basis vectors**. Basis vectors are independent and the number of basis vectors is called the **dimension** of the vector space.  
# 
# ## Rank and Decomposition of Matrices
# 
# The number of independent (row or column) vectors in a matrix is called its **rank**. If the number of independent vectors equals the number of columns (and rows), the matrix can be inverted. Matrix inversion can be conducted with the same techniques as solving linear equation systems. Matrix inversion can also be conducted with its determinant if it is different from zero implying a unique solution, and hereby, a unique inverse.
# 
# Furthermore, sometimes decomposition of matrices can be useful for certain applications. We will take a look at the **eigendecomposition** and the **singular value decomposition** before we end this subsection.
# 
# The eigen decomposition of a square matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ depends on **eigenvalues** and **eigenvectors** of if. If a number $\lambda \in \mathbb{R}$ exists such that:
# 
# $$ \boldsymbol{A} \boldsymbol{x} = \lambda \boldsymbol{x} $$
# 
# then we call $\lambda$ eigenvalue and $\boldsymbol{x} \in \mathbb{R}^n$ the eigenvector. As $\boldsymbol{A} \boldsymbol{0} = \lambda \boldsymbol{0}$ is always true, we are only interested in eigenvectors $\boldsymbol{x} \neq \boldsymbol{0}$.
# 
# We can rewrite the equation above to: 
# 
# $$ \left( \boldsymbol{A} - \lambda \boldsymbol{I}_n \right) \boldsymbol{x} = \boldsymbol{0}$$
# 
# One solution is always $\boldsymbol{x} = \boldsymbol{0}$ and as we are only interested in solutions different from this one, we are searching for values of $\lambda$ such that multiple solutions exist. For this to be true $\text{det}\left( \boldsymbol{A} - \lambda \boldsymbol{I}_n \right) = 0$ must hold, because if the determinant is not equal to zero only a unique solution exists which is $\boldsymbol{x} = \boldsymbol{0}$. Deriving the determinant in dependence of $\lambda$ leads to a $n$-th degree polynomial, the **characteristic polynomial**, whose roots are eigenvalues. Once we know those eigenvalues, we can derive corresponding eigenvectors with solving techniques of linear equation systems.
# 
# Given a $n \times n$ matrix has $n$ linearly independent eigenvectors with eigenvalues $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, ..., \lambda_n)$. The eigenvectors are concatenated column wise in the matrix:
# 
# $$
# \boldsymbol{X} =  
# \begin{pmatrix}
#    \boldsymbol{x}^{(1)} & \boldsymbol{x}^{(2)} & ... & \boldsymbol{x}^{(n)} \\
# \end{pmatrix} = 
# \begin{pmatrix}
#     x_{11} & \cdots & a_{1n}\\
#     x_{21} & \cdots & a_{2n}\\
#     \vdots &        & \vdots\\
#     x_{n1} & \cdots & x_{nn}
# \end{pmatrix}$$
# 
# then, the eigendecomposition of $\boldsymbol{A}$ is given by:
# 
# $$
# \boldsymbol{A} = \boldsymbol{X}\text{diag}\lbrace \lambda_i \rbrace \boldsymbol{X}^{-1}
# $$
# 
# Eigenvectors can be used for many things and are needed for principal component analysis which is a tool for dimensionality reduction in data analysis. Another interesting decomposition is the singular value decomposition which decomposes a matrix $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ into:
# 
# $$ 
# \boldsymbol{A} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{T}
# $$
# 
# Matrices $\boldsymbol{U}$ and $\boldsymbol{V}$ are **orthogonal** square matrices. An orthogonal matrix consists of mutually **orthonormal** vectors which means the pairwise vectors' dot products are equal to zero and the length of each vector is standardized to a norm of one. The matrix $\boldsymbol{D}$ is a diagonal matrix. The values along the diagonal are the **singular values**, while the columns of $\boldsymbol{U}$ are called **left-singular vectors** and the columns of $\boldsymbol{V}$ are called **right-singular vectors**. An example for an application of singular value decomposition is the generation of word embeddings, a method from the field of natural language processing, which is helpful to transform words into vectors. 

# # Analysis
# 
# ## Differential calculus
# One of the most important tools for us are derivatives of (continuous) functions. Especially, as they are very useful for an important task which is optimization.
# 
# The definition of a function's derivative is strongly related to the question to which degree the value of the function changes if its argument changes by a small amount. Typically, we are aware of derivatives for important functions such as: 
# 
# | $f(x)$ | $f'(x)$ |
# |:---|---:|
# | $a + bx$ | $b$ | 	
# | $x^p$ | $px^{p-1}$ | 
# | $e^x$ | $e^x$ |
# | $\ln x$ | $\frac{1}{x}$ |
# | $\sin x$ |$\cos x$ |
# | $\cos x$ |$-\sin x$|
# 
# and in combination with homogeneity, additivity, the product and chain rule:
# 
# * $\left(\lambda f\right)' = \lambda f'$
# * $\left(f + g\right)' = f' + g'$
# * $\left(f \cdot g\right)' = f'\cdot g + f \cdot g'$
# * $\left(f \circ g\right)' = \left(f' \circ g\right) \cdot g' = f'\left(g\left(x\right)\right)g'(x)$
# 
#  we are able to determine derivatives for a large family of composed functions. As you likely have heard before, the derivative $f'$ for a function $f: \mathbb{R} \to \mathbb{R}$ at a specific point $x_0$ can be interpreted as the slope of a tangent at $x_0$. Moreover, its value informs us about the monotonic behavior. A function is strictly monotonically increasing if $f'(x) > 0$ and strictly monotonically decreasing if $f'(x) < 0$.  Both can be examined in this figure: 

# In[1]:


import matplotlib.pylab as plt
import numpy as np

#definition of the functions
x_square = lambda x: x**2
x_square_neg = lambda x: - x**2

#definition of the derivatives
derivative = lambda x: 2 * x
derivative_neg = lambda x: -2 * x

x = np.linspace(-2, 5, 500)
epsilon = np.linspace(- 1, 1, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14,6))
ax1.plot(x, x_square(x))
ax1.set_xlabel('x')
ax1.set_ylabel(r'$f(x)$')
ax1.plot(np.linspace(2. - 1, 2. + 1, 200), np.array([x_square(2.) + i * derivative(2.) for i in epsilon]))

ax2.plot(x, x_square_neg(x))
ax2.set_xlabel('x')
ax2.set_ylabel(r'$f(x)$')
ax2.plot(np.linspace(2. - 1, 2. + 1, 200), np.array([x_square_neg(2.) + i * derivative_neg(2.) for i in epsilon]))

plt.show()


# ## Finding Extremes of Functions
# 
# Understanding the monotonic behavior of a function is very important in order to understand in which direction we need to go, if we want to reach the lowest or highest value of the function. For instance, in the plot on the left, the slope is positive at the value $2$, so even if we would not be able to see the plot, we know that we need to reduce the function's argument if we are seeking for lower function values and vice versa. Once we have reached a point for which $f'(x) = 0$, we reached a point at which the monotonic behavior potentially changes. Such a point is called a **stationary point**. A stationary point is either a **(local) minimum**, **(local) maximum** or a **saddle point**. Thus, finding stationary points is a necessary task to identify extremes, however, not sufficient as a saddle point is not an extreme value of a function.
# 
# To analyze what type of an extreme value we have found at a stationary point, the second derivative of a function can be used. The second derivative is informative regarding a function's curvature. Given a function $f:\mathbb{R} \to \mathbb{R}$, then $f$ is strictly convex, if for all $x, y \in \mathbb{R}, x \neq y$ with $\lambda \in (0, 1)$ it holds that:
# 
# $$ f(\lambda x + (1 - \lambda) y) < \lambda f(x) + (1 - \lambda) f(y) $$
# 
# and strictly concave if:
# 
# $$ f(\lambda x + (1 - \lambda) y) > \lambda f(x) + (1 - \lambda) f(y) $$
# 
# In the figure below, you see examples of a convex and a concave function. Imagine the tangent slopes of the first derivative for the convex function going from left to right. The slopes would strictly increase from negative to positive values, i.e., the first derivative of the first derivative (so the second derivative of the original function) is strictly increasing. Thus, the type of curvature can be evaluated with the second derivative. Second derivative values of strictly convex functions are strictly positive ($f''(x) > 0$) strictly negative ($f''(x) < 0$) for concave functions. As can be seen in the figure, a stationary point is a (local) minimum, if the function is convex and a (local) maximum if the function is concave.

# In[2]:


import matplotlib.pylab as plt

x = np.linspace(-2, 5, 50)
y = x**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))

ax1.plot(x, y)
ax1.plot([-2, 5], [4, 25])
ax1.text(-0.5, 3.5, 'convex')

ax2.plot(x, -y)
ax2.plot([-2, 5], [-4, -25])
ax2.text(-0.5, -3.5, 'concave')

plt.show()


# Summing up, the first derivative tells us in which direction we need to search for lower or higher values and once we reached a possible extreme value, the second derivative at this point tells us if we found a minimum (so if the function is convex) or a maximum (so if the function is concave).
# 
# Very often, we will face functions with multiple input arguments, meaning $f: \mathbb{R}^n \to \mathbb{R}$. For these functions, we need the **gradient** and the **Hessian** matrix to search for extremes of the function. The gradient of a function is given by
# 
# $$
# (\text{grad}) f (\boldsymbol{x}) = \nabla f (\boldsymbol{x}) = 
# \begin{pmatrix}
#     f_{x_1} (\boldsymbol{x}) \\
#     f_{x_2} (\boldsymbol{x}) \\
#     \vdots \\
#     f_{x_n} (\boldsymbol{x}) \\
# \end{pmatrix}
# $$
# 
# where, $\frac{\partial f}{\partial x_i} (\boldsymbol{x}) = f_{x_i} (\boldsymbol{x})$ stands for the **partial derivative** of the function $f$ with respect to $x_i$. Partial derivatives quantify the sensitivity of the function if $x_i$ changes, keeping all other variables constant. To derive the partial derivative, we treat the remaining variables like constant values and apply techniques that we apply to one dimensional functions. Partial derivatives of higher order can be derived in a similar fashion, whereby, the order after which variables we partially derive first is irrelevant. An important collection of partial derivatives is the collection of all second partial derivatives which are stored in the Hessian matrix $\boldsymbol{H} (\boldsymbol{x})$.
# 
# $$
# \boldsymbol{H} (\boldsymbol{x}) = 
# \begin{pmatrix}
#     f_{x_1 x_1} (\boldsymbol{x}) & f_{x_1 x_2} (\boldsymbol{x}) & ... & f_{x_1 x_n} (\boldsymbol{x}) \\
#     f_{x_2 x_1} (\boldsymbol{x}) & f_{x_2 x_2} (\boldsymbol{x}) & ... & f_{x_2 x_n} (\boldsymbol{x})\\
#     \vdots  & \vdots & & \vdots \\
#     f_{x_n x_1} (\boldsymbol{x}) & f_{x_n x_2} (\boldsymbol{x}) & ... & f_{x_n x_n} (\boldsymbol{x})\\
# \end{pmatrix}
# $$
# 
# For a stationary point, 
# 
# $$ 
# \nabla f (\boldsymbol{x}) = \boldsymbol{0}
# $$
# 
# holds. If the Hessian matrix is positive definite (all its eigenvalues are positive) at this point, we found a local minimum. If it is negative definite (all its eigenvectors) are negative, we found a local maximum.
# 
# 
# Mainly, we will use these techniques to optimize functions at a later stage. Optimization in our case typically means we are looking for the best possible way to adjust our data model to observed data. Hereby, we measure the level of adjustment with the help of **objective functions** which are also often called **loss functions** in this context. 

# # Statistics
# 
# Why do we need statistics for data analytics? We observe randomness in more or less all fields of economics. For instance, spending a certain amount of money to promote a product will not increase its sales by a deterministic amount. Or if good news about a company are shared, we do not know for certain how this affects its future stock price. This causes uncertainty for which we use probability theory and its methods. 
# 
# ## Probability Theory
# 
# To express uncertainty via probabilities, we need three things:
# 
# 1. A set of all possible outcomes $\Omega = \{ \omega_1, ..., \omega_n  \}$
# 2. **Events** which are subsets $A \subseteq \Omega$ and a set of events $\mathcal{F}$ called $\sigma$-algebra, which fulfills three requirements:
#     1. $\Omega \in \mathcal{F}$
#     2. $A \in \mathcal{F} \Rightarrow \bar{A} \in \mathcal{F}$
#     3. $A_1, A_2, ... \in \mathcal{F} \Rightarrow \bigcup\limits_{i \in \mathbb{N}} A_i \in \mathcal{F}$
# 3. A mapping called **probability** $P: \mathcal{F} \rightarrow [0,1]$  which assigns numbers $[0,1]$ to every event $A$. The probability mapping must fulfill the axioms of probability:
#     1. $P(A \geq 0)~ \forall A_i~ \in \mathcal{F}$
#     2. $P(\Omega) = 1$
#     3. $P\left(\bigcup\limits_{i = 1}^{\infty} A_i \right) = \sum_{i = 1}^{\infty} P(A_i) \forall A_i \in \mathcal{F}$ with $A_j \cap A_k = \emptyset$, $j \neq k$
#     
# Usually, we are not interested in the event itself but in a number associated with the event, e.g., units sold, revenue, price, ... We call these numbers **random numbers** which are realizations of **random variables**. A one-dimensional random variable $X$ maps the set of outcomes $\Omega$ to real numbers:
# 
# $$X: \Omega \to \mathbb{R}$$
# $$\omega \to X(\omega)$$
# 
# For instance a coin tossing game with outcomes $\omega_1$: *head* and $\omega_2$: *tail*. If *head* is the outcome, we win 5 bucks and if *tail* is the outcome we loose 5 bucks. The random number $X$ describing our profit for a coin toss is defined by:
# 
# $X(\omega_1) = 5$ and $X(\omega_2) = -5$. More technically speaking $X$ must be **measurable**. Given a subset of $B \in \mathbb{R}$, the inverse image of $B$ with respect to $X$ is:
# 
# $$X^{-1}(B) = \{\omega \in \Omega | X(\omega) \in B$$
# 
# Let $\mathcal{B}$ be a set of subsets from $\mathbb{R}$, $X$ is measurable if $X^{-1}(B) \in \mathcal{F}$ for all $B \in \mathcal{B}$. For the probability measure of $X$, it holds that:
# 
# $$P_X(X \in B) = P(X^{-1}(B)), B \in \mathcal{B}$$
# 
# Thus, $X$ is not just a measure mapping outcomes to numbers, but rather maps from one probability space $(\Omega, \mathcal{F}, P)$ to probability space $(\mathbb{R}, \mathcal{B}, P_X)$. In the following, we use capital letters for random variables and lower case letters for concrete realizations of the random variable, e.g., $P(X = x) = P(X = 5) = 0.5$. For a vector of random variables we write $\boldsymbol{X} = (X_1, ..., X_n)$ and for a concrete realization we write $\boldsymbol{x} = (x_1, ..., x_n)$.
# 
# ## Univariate Random Variables
# 
# Random variables can be **discrete** or **continuous**. The former has only a finite amount of realizations (or infinite countable realizations), while the latter can exhibit an infinite amount of different realizations. 
# 
# For a discrete random variable, the function assigning probabilities to its realizations is called **probability mass function** and is given by a list-alike definition:
# 
# $$
# f(x) = \begin{cases}
# 			P_X(\{ x \}) & \text{for } X = x  \\
# 			0 & \, \text{else}
#          \end{cases}
# $$
# 
# 
# The function 
# 
# $$
# F(x) = P_X\left( \{X |X \leq x \} \right) = P_X\left( (-\infty, x] \right) = \sum_{x_i \leq x} f(x_i)
# $$
# 
# is called **cumulative distribution function**.
# 
# For a continuous random variable, the probability of a concrete realization is equal to 0, $P(X = x) = 0$. Instead of directly defining a function mapping probabilities to subsets of $\mathbb{R}$, we define a **probability density function** $f(x)$ which enables us to determine probabilities by integration. The domain of this function must be the set of all possibles states of $X$. Furthermore, it must hold the $f(x) \geq 0$ for all $x$ and that $\int_{-\infty}^{\infty} f(x)dx = 1$. With this definition the corresponding cumulative distribution function is given by:
# 
# $$
# F(x) = \int_{-\infty}^{x} f(t)dt
# $$
# 
# Probability distributions can be compared by numbers which summarize their characteristics such as location, variation, shape and so on. For this purpose, ordinary and central moments of the distributions are used. 
# 
# The **expectation** or **expected value** of a random variable if defined as:
# 
# $$
# E(X) = \begin{cases}
# 			\sum_i x_i \cdot f(x_i) &  X   \text{ discrete} \\
# 			\int_{-\infty}^{\infty} x \cdot f(x) dx & X \text{ continuous}
#        \end{cases}
# $$
# 
# The expectation is linear which means: $E(X + Y) = E(X) + E(Y)$ and $E(aX) = a E(X)$ holds. In addition the relation $E(a + bX) = a + b E(X)$ can be useful in some cases.
# 
# The **variance** measures how much realizations of a random variable vary and is defined by:
# 
# $$
# Var(X) = E(X-E(X))^2 =
# 			\begin{cases}
# 			\sum_i (x_i - E(X))^2 \cdot f(x_i)   & X \text{ discrete} \\
# 			\int_{-\infty}^{\infty} (x - E(X))^2 \cdot f(x) dx  & X \text{ continuous}
# 			\end{cases}
# $$
# 
# For the variance, it holds that $Var(a + bX) = b^2 Var(X)$. The square root of the variance defines the **standard deviation** which is often denoted by the symbol $\sigma_X$.
# 
# If a probability distribution is not symmetric it is skewed. To quantify **skewness**, we use:
# 
# $$
# \frac{E \left[\left(X - E(X)\right)^3 \right]}{\sigma_X^3}  
# $$
# 
# For negative values, the distribution is left skewed and for positive values, the distribution is right skewed. In addition of skewness, **kurtosis** is also often used for the characterization of probability distributions. 
# 
# $$
# \frac{E \left[\left(X - E(X)\right)^4 \right]}{\sigma_X^4}  
# $$
# 
# To characterize the level of kurtosis, the normal distribution is usually taken as a reference. It exhibits a kurtosis of $3$. If distributions exhibit higher kurtosis than $3$, we speak of excess kurtosis and leptokurtic distributions. These distributions are characterized by higher probability mass in the tails of the distribution. This means, extreme outcomes of the random variable are more likely. If a distribution exhibits a kurtosis lower then $3$, we say it is platokurtic. 
# 
# Besides those four characteristic measures, quantiles of probability distributions can be used for informational purposes. Let us denote $F^{-1}: [0, 1] \to \mathbb{R}$ as the inverse of the cumulative probability density function. Given some probability $\alpha$, we say $x_{\alpha}$ is the $\alpha$ **quantile** of $X$, if:
# 
# $$
# 1 - F(x_{\alpha}) \geq 1 - \alpha ~~\text{and } F(x_{\alpha}) \geq \alpha
# $$
# 
# For instance assume you own an insurance company and $X$ represents losses in million due to damages of houses for a year. A value of $x_{0.95} = 1000$ tells us that losses higher than 1000 will only be exceeded with a probability of 95%. If another company has a value which is higher, e.g., 1100, we immediately know that the latter company faces a greater probability of high losses. At the same time it is possible that both companies have the same expectation for such losses. 
# 
# ## Important Distributions
# 
# There is a multitude of interesting and useful probability distributions. However, we only name a few examples which are most relevant for data modeling. 
# 
# **Bernoulli Distribution**
# 
# The Bernoulli distribution is the distribution of a random variable with two outcomes $X = 0$ and $X = 1$. The distribution is defined by a single parameter $\pi \in [0, 1]$ which equals the probability for $P(X = 1) = \pi$. As we only have two outcomes this also defines the probability $P(X = 0) = 1 - P(X = 1) = 1 - \pi$. Accordingly, the expectation is given by:
# 
# $$
# E(X) = \pi \cdot 1 + (1 - \pi) \cdot 0 = \pi
# $$
# 
# and the variance is:
# 
# $$
# Var(X) = \pi \cdot (1 - \pi)^2 + (1 - \pi) \cdot (0 - \pi)^2 = \pi (1 - \pi)(1 - \pi + \pi) = \pi (1 - \pi)
# $$
# 
# **Categorical Distribution**
# 
# If a random variable has $K$ possible realizations, we call its distribution categorical of multinoulli. Let $\boldsymbol{p} = (p_1, ..., p_K)$ define the probability vector where each single entry $p_k$ represents the probability $P(X = k)$ and $\sum_{k = 1}^{K} p_k = 1$. Accordingly, the probability mass function can be written by:
# 
# $$
# f(x) = \prod_{k = 1}^{K} p_k^{[x = k]}
# $$
# 
# with 
# 
# $$
# [x = k] = \begin{cases}
#             1 & \text{ if } x = k \\
#             0 & \text{ else}
# 		  \end{cases}
# $$
# 
# Typically the expectation or variance of the categorical distribution is not of great interest to us because in most of the times the numbers $k$ stand for different categories which are not represented by the specific value of $X$.
# 
# **Normal Distribution**
# The normal distribution is often used for various applications. It is defined by two parameters $\mu, \sigma > 0$ and can be used for continuous variables over $\mathbb{R}$. Its density function is given by:
# 
# $$
# f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{\left( x - \mu \right)^2}{2 \sigma^2} \right)
# $$
# 
# The normal distribution is able to mimic real-life phenomena like measurement errors, but its frequent usage is probably better explainable by its mathematical properties which often make results analytically traceable. 
# 
# ## Multiple Random Variables
# 
# Typically, we will take a look at multiple random variables at the same time. Very often certain relationships exist between these variables. The distribution over multiple random variables is described by their univariate characteristics together with their linkages:
# 
# $$
# F(\boldsymbol{x}) = F(x_1, ..., x_n) = P(X_1 \leq x_1, ..., X_n \leq x_n)
# $$
# 
# Technically $F$ can be separated into $F(x_1, ..., x_n) = C(F_{X_1}(x_1), ..., F_{X_n}(x_n))$ where $C$ is called a copula which specifies the dependence structure between the univariate random variables and $F_{X_1}, ..., F_{X_n}$ are univariate distributions.
# 
# Most important to us are the concepts of independence, dependence and conditionality in the context of multivarite distributions. 
# 
# The **conditional distribution** of two random variables is defined by:
# 
# $$
# f_{X_1 | X_2}(x_1 | x_2) = \frac{f_{X_1, X_2}(x_1, x_2)}{f_{X_2}(x_2)}
# $$
# 
# We can abstract this definition and derive the **chain rule of conditional probabilities**. Assume we have:
# 
# $$
# f_{X_1, X_2, X_3}(x_1, x_2, x_3) = f_{X_1 | X_2, X_3}(x_1 | x_2, x_3) f_{X_2, X_3}(x_2, x_3)
# $$
# 
# $$
# f_{X_2, X_3}(x_2, x_3) = f_{X_2 | X_3}(x_2 | x_3) f_{X_3}(x_3)
# $$
# 
# combining this, leads to:
# 
# $$
# f_{X_1, X_2, X_3}(x_1, x_2, x_3) = f_{X_1 | X_2, X_3}(x_1 | x_2, x_3) f_{X_2 | X_3}(x_2 | x_3) f_{X_3}(x_3)
# $$
# 
# The generalization leads to:
# 
# $$
# f_{X_1, ..., X_n}(x_1, ..., x_n) = f_{X_1}(x_1) \prod_{i = 2}^{n} f_{X_i | X_1, ..., X_{i-1}}(x_i | x_1, ..., x_{i-1})
# $$
# 
# 
# Random variables $X_1, ..., X_n$ are **independent** if their joint distribution is given by the product of their univariate distributions:
# 
# $$
# f(x_1, ...., x_n) = f_{X_1}(x_1) \cdot ... \cdot f_{X_n}(x_n)
# $$
# 
# We will often find ourselves in the position in which we are interested in conditional point of views. For instance, what is the probability for granting a credit to a person, if the person is male and below 25 years old. Conditional distributions are defined for discrete as well as continuous product set. In the discrete case the **conditional probability mass function** for two random variables is defined as:
# 
# $$
# f_{X_1 | X_2}(x_1 | x_2) = \frac{f_{X_1, X_2}(x_1, x_2)}{f_{X_2}(x_2)}
# $$
# 
# for a fixed realization $x_2$ and all $x_1$ in the domain of $X$. The corresponding **conditional cumulative distribution** function is given by:
# 
# $$
# F_{X_1 | X_2}(x_1 | X_2 = x_2) = \sum_{x_{1i} \leq x_1} f_{X_1 | X_2}(x_{1i} | X_2 = x_2) = \sum_{x_{1i} \leq x_1} \frac{f_{X_1, X_2}(x_{1i}, x_2)}{f_{X_2}(x_2)}
# $$
# 
# The **conditional probability density function** for continuous variables is also given by:
# 
# $$
# f_{X_1 | X_2}(x_1 | x_2) = \frac{f_{X_1, X_2}(x_1, x_2)}{f_{X_2}(x_2)}
# $$
# 
# while the **conditional cumulative distribution function** is defined as:
# 
# $$
# F_{X_1 | X_2}(x_1 | X_2 = x_2) = \int_{-\infty}^{x_1} \frac{f_{X_1, X_2}(u,x_2)}{f_{X_2}(x_2)} \, du
# $$
# 
# Also of importance to us is the concept of **conditional independence**. Often, we will find ourselves in the situation in which we use some predictor variables $\boldsymbol{X}$ to learn something about a target variable $Y$. When we estimate parameters for models which define a relationship between $\boldsymbol{X}$ and $Y$, we often implicitly assume that realizations of the target variables are independent, conditional on the realizations of predictor variables. 
# 
# In general, two random variables $Y_1, Y_2$ are conditionally independent, given the realization of a random variable $X$ if:
# 
# $$
# f_{Y_1, Y_2 | X = x}(y_1, y_2) = f_{Y_1 | X = x}(y_1) f_{Y_2 | X = x}(y_2) 
# $$
# 
# This means, conditional on the observation $x$, what happens with $Y_1$ has no impact on the observation of $Y_2$ and vice versa. Note that we will even assume a bit more when we estimate parameters for our models. That is, given realizations of $X_i$, $i = 1, ..., n$, we assume all $Y_i$ to be conditionally independent, so, e.g. for two random variables:
# 
# $$
# f_{Y_1, Y_2 | X_1 = x_1, X_2 = x_2}(y_1, y_2) = f_{Y_1 | X_1 = x_1}(y_1) f_{Y_2 | X_2 = x_2}(y_2) 
# $$
# 
# To quantify **linear dependence** between two random variables the **covariance** is used.
# 
# $$
# Cov(X,Y) = E\left[(X_1-E(X_1))\cdot (X_2-E(X_2)) \right] = E(X_1\cdot X_2) - E(X_1)E(X_2)
# $$
# 
# The covariance can be scaled by the standard deviations of $X_1, X_2$ which leads to the **coefficient of linear correlation** which is in the range $[-1, 1]$:
# 
# $$
# \rho = \frac{Cov(X_1,X_2)}{\sqrt{Var(X_1)} \cdot \sqrt{Var(X_2)}}
# $$
# 
# It is important to understand that only linear dependence is captured adequately by covariance. Therefore, it is sometimes more reasonable to take into account metrics which also capture non-linear dependencies in an adequate way. An example is Spearman's rho which is defined by:
# 
# $$
# \rho_S = \frac{Cov(rg_{X_1},rg_{X_2})}{ \sigma_{rg_{X_1}} \cdot \sigma_{rg_{X_2}}}
# $$
# 
# where $rg$ stands for the rank of the realization and not its value.
# 
# ## Parameter Estimation
# 
# In general, we will often be interested in finding a model for random variables $Y_i$, $i = 1, ..., n$. Most of the times, we try to improve our understanding of $Y_i$ by including predictor variables $\boldsymbol{X}_i$. With and without predictor variables, a model for $Y_i$ is typically **parametric** which means the model specification depends on parameters $\boldsymbol{\theta}$. An example is the normal distribution which depends on two parameters or a linear regression model which includes parameters for the regression line and the variance of the error variable.
# 
# We will come back to parameter estimation in detail in subsequent chapters. In general, three ways of estimation are discussed for parameter estimation:
# 
# 1. Least-Squares Estimation
# 2. Maximum-Likelihood Estimation
# 3. Bayesian Estimation
# 
# **Least-Squares Estimation**
# 
# Least-squares estimation refers to a minimization problem. The unknown parameter is derived via minimizing squared deviations between the random variable and the parameter:
# 
# $$
# \min_{\theta} \sum_{i = 1}^{n} (Y_i - \theta)^2
# $$
# 
# Even though this is a general concept, it is mostly used for estimation of expectations or conditional expectations.
# 
# **Maximum-likelihood estimation**
# 
# Maximum-likelihood estimation refers to a maximization problem. The unknown parameter is derived via maximizing the likelihood of the data $(Y_1, ..., Y_n)$. This presupposes a certain distributional form for $Y_i$ which is expressed by the probability mass or density function $f$ which depends on parameter values. Assuming independent and identically distributed (iid) random variables, the likelihood of the data is given by:
# 
# $$
# \mathcal{L}(\boldsymbol{\theta}) = \prod_{i = 1}^{n} f(Y_i | \boldsymbol{\theta})
# $$
# 
# Out of technical reasons, we usually maximize the log-likelihood:
# 
# $$
# \ln\left( \mathcal{L}(\boldsymbol{\theta}) \right) = \sum_{i = 1}^{n} \ln \left(f(Y_i | \boldsymbol{\theta})\right)
# $$
# 
# The parameter is estimated by:
# 
# $$
# \boldsymbol{\theta}_{ML} = \arg \max_{\boldsymbol{\theta}} \ln\left( \mathcal{L}(\boldsymbol{\theta}) \right)
# $$
# 
# **Bayesian estimation**
# 
# Bayesian estimation refers to a problem which differs in its logic to the previous two approaches. To express uncertainty about the unknown parameter, it is described by a probability distribution. The aim is to estimate the parameter's distribution after observing data $\boldsymbol{y} = (y_1, ..., y_n)$, i.e., $f(\theta | \boldsymbol{y})$. The estimate itself is taken from this distribution and usually we use the distribution's expectation or median. The distribution is derived using Bayes theorem:
# 
# $$
# f(\theta | \boldsymbol{y}) = \frac{f(\boldsymbol{y} | \theta) f(\theta)}{\int f(\boldsymbol{y} | \theta)f(\theta) d\theta}
# $$
# 
# $f(\boldsymbol{y} | \theta)$ represents the likelihood of the data and $f(\theta)$ is the prior distribution which reflects our idea about the unknown parameter before we observe any data.
# 
# ## Bias and Variance
# 
# Parameters are estimated on $m$ random samples which makes the estimator $\theta_m$ a random number itself. If the expected value of the estimator equals its true value, we say the estimator is **unbiased**. On the contrary, the **bias** can be defined by:
# 
# $$
# \text{bias}(\hat{\theta}_m) = E(\hat{\theta}_m) - \theta
# $$
# 
# You may picture it like this: Assume you can draw many random samples from the data generating process. Due to randomness your parameter estimate will differ for different samples. However after a very large amount of random samples the average of all parameter estimates will be equal to the true parameter. Sometimes this is only true if the random samples are of increasing size. In case of the latter, we speak of parameters which are **asymptotically unbiased** and write $\lim \limits_{m \to \infty} E(\hat{\theta}_m) = \theta$.
# 
# For obvious reasons, unbiasedness is a pleasant attribute of an estimator, but not the only one we care about. Besides, we want our estimates not to vary too much for different data sets. Therefore, we also take a look at the variance of an estimator $Var(\hat{\theta}_m)$. As both, bias and variance are important attributes, we usually use the **mean squared error** to evaluate and compare estimators. The mean squared error is given by:
# 
# $$
# \text{MSE} = E \left[ (\hat{\theta}_m - \theta)^2\right] = \text{bias}(\hat{\theta}_m)^2 + Var(\hat{\theta}_m)
# $$
