# Deriving the Exponential of a 2x2 Matrix

Suppose the matrix \( A \) given is:
  
$$
A = \begin{pmatrix}
a & b \\
-b & a
\end{pmatrix}
$$

The exponential \( e^{At} \) is expressed as:
  
$$
e^{At} = e^{at} \begin{pmatrix}
\cos bt & \sin bt \\
-\sin bt & \cos bt
\end{pmatrix}
$$

This expression suggests a complex number exponentiation, considering \( A \) as a matrix representing a rotation-scaling transformation in the complex plane. Let's go through the steps to derive this result analytically.

### Eigenvalues and Eigenvectors

To compute \( e^{At} \), first find the eigenvalues and eigenvectors of \( A \). The eigenvalues \( \lambda \) of \( A \) can be found by solving the characteristic equation:
  
$$
\det(A - \lambda I) = 0
$$

For the given \( A \), the characteristic equation is:
  
$$
\det\begin{pmatrix}
a - \lambda & b \\
-b & a - \lambda
\end{pmatrix} 
= (a-\lambda)^2 + b^2 = 0
$$

Solving this yields the complex eigenvalues:
  
$$
\lambda = a \pm bi
$$

### Matrix Exponential using Eigenvalues

The matrix \( A \) is similar to a matrix \( B \) in Jordan form, which in this case, due to the distinct eigenvalues \( a \pm bi \), is diagonal:
  
$$
B = \begin{pmatrix}
a+bi & 0 \\
0 & a-bi
\end{pmatrix}
$$

The matrix exponential of \( B \) is straightforward to compute:
  
$$
e^{Bt} = \begin{pmatrix}
e^{(a+bi)t} & 0 \\
0 & e^{(a-bi)t}
\end{pmatrix}
$$

Using Euler's formula \( e^{ix} = \cos x + i \sin x \), we have:
  
$$
\begin{aligned}
e^{(a+bi)t} &= e^{at}(\cos bt + i\sin bt) \\
e^{(a-bi)t} &= e^{at}(\cos bt - i\sin bt)
\end{aligned}
$$

Therefore,
  
$$
e^{Bt} = \begin{pmatrix}
e^{at} \cos bt & e^{at} \sin bt \\
-e^{at} \sin bt & e^{at} \cos bt
\end{pmatrix}
$$

### Relating \( B \) to \( A \)

Since \( A \) and \( B \) are similar, there exists a matrix \( P \) such that \( A = PBP^{-1} \). In this case, \( P \) can be constructed from the eigenvectors of \( A \). However, for this specific \( A \), which is a standard form of a complex rotation matrix, the matrix \( P \) and its inverse would yield the direct transformation between the Jordan form and the given \( A \), maintaining the same exponential result.

### Linking to Complex Numbers

For the transformation seen in \( e^{At} \), where the elements involving \( i\sin(bt) \) seem to cross over to another column (or row), consider that complex exponentials decompose into their real and imaginary parts. The real parts \( \cos(bt) \) remain on the diagonal, maintaining the direct influence of the real part of the eigenvalues, while the imaginary parts (which would include \( i\sin(bt) \)) manifest as the sine functions on the off-diagonals, with one positive and one negative to align with rotational symmetry in two dimensions.


To address the conditions under which a 2x2 matrix with complex diagonal entries can have its complex values moved to another column, we need to consider the concept of similarity transformations and diagonalization for matrices with complex eigenvalues.

#### Conditions for Moving Complex Values

A 2x2 matrix with complex diagonal entries can have its complex values moved to another column under the following conditions:

1. The matrix has complex conjugate eigenvalues.
2. The matrix is diagonalizable.

Let's explore these conditions in more detail:

#### Complex Conjugate Eigenvalues

For a 2x2 matrix A with complex diagonal entries, it must have eigenvalues that are complex conjugates of each other. This means the eigenvalues are in the form a ± bi, where a and b are real numbers and i is the imaginary unit[1][3].

#### Diagonalizability

The matrix must be diagonalizable. This means there exists an invertible matrix P such that P^(-1)AP is a diagonal matrix[1][3]. For 2x2 matrices with distinct complex eigenvalues, diagonalizability is guaranteed.

#### Similarity to a Rotation-Scaling Matrix

When these conditions are met, the matrix A is similar to a rotation-scaling matrix of the form:

$$
C = \begin{bmatrix}
a & -b \\
b & a
\end{bmatrix}
$$

where a and b are the real and imaginary parts of the eigenvalue, respectively[12].

#### Transformation Process

The process of moving the complex values to another column involves finding a similarity transformation that converts the original matrix to this rotation-scaling form. This is achieved through the following steps:

1. Find the complex eigenvalues λ = a ± bi.
2. Compute the corresponding eigenvectors.
3. Form a matrix P with columns as the real and imaginary parts of an eigenvector.
4. The similarity transformation P^(-1)AP will result in the rotation-scaling matrix C[12].

#### Example

Consider a matrix A with complex diagonal entries:

$$
A = \begin{bmatrix}
3+i & 0 \\
0 & 3-i
\end{bmatrix}
$$

This matrix has eigenvalues 3±i. We can find a similarity transformation to convert it to:

$$
C = \begin{bmatrix}
3 & 1 \\
-1 & 3
\end{bmatrix}
$$

The matrix P that achieves this transformation would have columns representing the real and imaginary parts of an eigenvector of A.



### Conclusion

Thus,
  
$$
e^{At} = e^{at} \begin{pmatrix}
\cos bt & \sin bt \\
-\sin bt & \cos bt
\end{pmatrix}
$$

This result corresponds to the matrix undergoing a combined rotation and scaling transformation, where \( e^{at} \) scales the magnitude, and the \( 2 \times 2 \) rotation matrix 
$$
\begin{pmatrix}
\cos bt & \sin bt \\
-\sin bt & \cos bt
\end{pmatrix}
$$ 
rotates the space by angle \( bt \).
