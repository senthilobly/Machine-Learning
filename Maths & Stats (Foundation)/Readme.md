# Complete Linear Algebra Essentials for Machine Learning

---

## I. Data Types & Structures

| Type                         | Description                           | Examples               |
| ---------------------------- | ------------------------------------- | ---------------------- |
| **Scalar**                   | Single number (0D)                    | `x = 5`                |
| **Vector**                   | 1D array of numbers                   | `v = [1, 2, 3]`        |
| **Matrix**                   | 2D array of numbers                   | `A = [[1, 2], [3, 4]]` |
| **Tensor**                   | ND array (3D or more)                 | `T = image batch`      |
| **Row Vector**               | 1 × n matrix                          | `[1, 2, 3]`            |
| **Column Vector**            | n × 1 matrix                          | `[[1], [2], [3]]`      |
| **Sparse Matrix**            | Matrix with mostly zero entries       | Used in NLP, Graphs    |
| **Diagonal Matrix**          | Non-zero only on diagonal             | `diag([1, 2, 3])`      |
| **Identity Matrix**          | Diagonal of ones                      | `I = eye(n)`           |
| **Symmetric Matrix**         | Equal to its transpose                | Covariance matrix      |
| **Orthogonal Matrix**        | Transpose is inverse                  | QR decomposition       |
| **Positive Definite Matrix** | x^T A x > 0                           | Optimization, SVM      |
| **Projection Matrix**        | Projects vector to subspace           | Least squares, PCA     |
| **Rank**                     | No. of linearly independent rows/cols | Dimensionality measure |

---

## II. Core Operations

| Operation                 | Description                                 |   |   |   |    |
| ------------------------- | ------------------------------------------- | - | - | - | -- |
| **Addition/Subtraction**  | Element-wise operations on vectors/matrices |   |   |   |    |
| **Scalar Multiplication** | Multiply each element by a number           |   |   |   |    |
| **Matrix Multiplication** | Dot product or linear transformation        |   |   |   |    |
| **Dot Product (Inner)**   | `a . b = sum(a_i * b_i)`                    |   |   |   |    |
| **Outer Product**         | `a ⊗ b = a * b^T`                           |   |   |   |    |
| **Hadamard Product**      | Element-wise product                        |   |   |   |    |
| **Transpose**             | Flipping a matrix over its diagonal `A^T`   |   |   |   |    |
| **Inverse**               | `A^-1 * A = I`                              |   |   |   |    |
| **Determinant**           | Scalar measure of matrix volume/scaling     |   |   |   |    |
| **Norm**                  | Length of a vector \`                       |   | x |   | \` |
| **Trace**                 | Sum of diagonal elements `tr(A)`            |   |   |   |    |
| **Rank**                  | No. of independent rows/columns             |   |   |   |    |
| **Diagonalization**       | Convert matrix to diagonal form             |   |   |   |    |

---

## III. Important Concepts & Methods

| Concept/Method                         | Application in ML                                  |
| -------------------------------------- | -------------------------------------------------- |
| **Linear Independence**                | Feature selection, matrix rank                     |
| **Span**                               | Linear combinations and subspaces                  |
| **Basis & Dimension**                  | Define feature space                               |
| **Orthogonality**                      | Uncorrelated features                              |
| **Eigenvalues & Eigenvectors**         | PCA, stability analysis                            |
| **Spectral Decomposition**             | Matrix understanding via eigenvalues               |
| **Singular Value Decomposition (SVD)** | Dimensionality reduction                           |
| **QR Decomposition**                   | Numerical solutions, Gram-Schmidt process          |
| **LU Decomposition**                   | System of equations                                |
| **Cholesky Decomposition**             | Fast optimization when matrix is positive definite |
| **Moore–Penrose Pseudoinverse**        | Solving non-invertible systems (linear regression) |
| **Gradient (as Vector)**               | Optimization direction in ML                       |
| **Hessian (as Matrix)**                | Second-order derivatives in optimization           |
| **Projection**                         | Least squares, PCA                                 |
| **Covariance Matrix**                  | Statistical relationships, PCA                     |
| **Gram Matrix**                        | Kernel trick in SVMs                               |

---

## IV. Applications in ML

| ML Area                         | Linear Algebra Usage                        |
| ------------------------------- | ------------------------------------------- |
| **Linear Regression**           | Normal equations, pseudoinverse, projection |
| **Logistic Regression**         | Vectorized sigmoid and gradient operations  |
| **PCA**                         | Eigenvectors of covariance matrix           |
| **Neural Networks**             | Tensor operations, matrix multiplications   |
| **SVM**                         | Dot product, kernel trick (Gram matrix)     |
| **Clustering (K-Means)**        | Euclidean distances (vector norms)          |
| **Recommendation Systems**      | Matrix factorization (SVD)                  |
| **Natural Language Processing** | Word embeddings, sparse matrices            |
| **Optimization**                | Gradients, Hessians, eigen analysis         |

---

## Summary Overview

| Category           | Key Components                                                           |
| ------------------ | ------------------------------------------------------------------------ |
| **Data Types**     | Scalar, Vector, Matrix, Tensor, Sparse, Identity, Diagonal, etc.         |
| **Operations**     | Addition, Multiplication, Transpose, Inverse, Norm, Dot/Hadamard Product |
| **Concepts**       | Linear independence, Eigenvalues, Orthogonality, Rank, Projection        |
| **Decompositions** | SVD, Eigen, QR, LU, Cholesky, Pseudoinverse                              |
| **Applications**   | Linear Models, PCA, SVM, NN, Optimization, NLP, Recommender Systems      |

---

