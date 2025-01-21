For example, suppose the matrix \( A \) given is:
\[
A = \begin{bmatrix} a & b \\ -b & a \end{bmatrix}
\]
The exponential \( e^{At} \) is expressed as:
\[
e^{At} = e^{at} \begin{bmatrix} \cos bt & \sin bt \\ -\sin bt & \cos bt \end{bmatrix}
\]
This expression suggests a complex number exponentiation, considering \( A \) as a matrix representing a rotation-scaling transformation in the complex plane. Let's go through the steps to derive this result analytically.

### Eigenvalues and Eigenvectors
To compute \( e^{At} \), first find the eigenvalues and eigenvectors of \( A \). The eigenvalues \( \lambda \) of \( A \) can be found by solving the characteristic equation:
\[
\det(A - \lambda I) = 0
\]
For the given \( A \), the characteristic equation is:
\[
\det\left(\begin{bmatrix} a - \lambda & b \\ -b & a - \lambda \end{bmatrix}\right) = (a-\lambda)^2 + b^2 = 0
\]
Solving this yields the complex eigenvalues:
\[
\lambda = a \pm bi
\]

### Matrix Exponential using Eigenvalues
The matrix \( A \) is similar to a matrix \( B \) in Jordan form, which in this case, due to the distinct eigenvalues \( a \pm bi \), is diagonal:
\[
B = \begin{bmatrix} a+bi & 0 \\ 0 & a-bi \end{bmatrix}
\]
The matrix exponential of \( B \) is straightforward to compute:
\[
e^{Bt} = \begin{bmatrix} e^{(a+bi)t} & 0 \\ 0 & e^{(a-bi)t} \end{bmatrix}
\]
Using Euler's formula \( e^{ix} = \cos x + i \sin x \), we have:
\[
e^{(a+bi)t} = e^{at}(\cos bt + i\sin bt) \\
e^{(a-bi)t} = e^{at}(\cos bt - i\sin bt)
\]
Therefore,
\[
e^{Bt} = \begin{bmatrix} e^{at} \cos bt & e^{at} \sin bt \\ -e^{at} \sin bt & e^{at} \cos bt \end{bmatrix}
\]

### Relating \( B \) to \( A \)
Since \( A \) and \( B \) are similar, there exists a matrix \( P \) such that \( A = PBP^{-1} \). In this case, \( P \) can be constructed from the eigenvectors of \( A \). However, for this specific \( A \), which is a standard form of a complex rotation matrix, the matrix \( P \) and its inverse would yield the direct transformation between the Jordan form and the given \( A \), maintaining the same exponential result.

### Conclusion
Thus, \( e^{At} \) is:
\[
e^{At} = e^{at} \begin{bmatrix} \cos bt & \sin bt \\ -\sin bt & \cos bt \end{bmatrix}
\]
This result corresponds to the matrix undergoing a combined rotation and scaling transformation, where \( e^{at} \) scales the magnitude, and the 2x2 rotation matrix \( \begin{bmatrix} \cos bt & \sin bt \\ -\sin bt & \cos bt \end{bmatrix} \) rotates the space by angle \( bt \).
