from __future__ import absolute_import
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace, dot
from . import numpy_wrapper as anp

wrap_namespace(npla.__dict__, globals())

def atleast_2d_col(x):
    # Promotes a 1D array into a column rather than a row.
    return x if x.ndim > 1 else x[:,None]

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
inv.defgrad(    lambda ans, x    : lambda g : -dot(dot(ans.T, g), ans.T))
det.defgrad(    lambda ans, x    : lambda g : g * ans * inv(x).T)
slogdet.defgrad(lambda ans, x    : lambda g : g[1] * inv(x).T)
solve.defgrad(  lambda ans, a, b : lambda g : -dot(atleast_2d_col(solve(a.T, g)),
                                                 atleast_2d_col(ans).T))
solve.defgrad(lambda ans, a, b : lambda g : solve(a.T, g), argnum=1)
norm.defgrad( lambda ans, a    : lambda g : dot(g, a/ans))

def make_grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[0]
    w, v = ans              # Eigenvalues, eigenvectors.
    def eigh_grad(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = anp.repeat(w[:, anp.newaxis], N, 1)
        off_diag = anp.ones((N, N)) - anp.eye(N)
        F = off_diag / (w_repeated.T - w_repeated + anp.eye(N))
        dx = dot(v * wg + dot(v, F * dot(v.T, vg)), v.T)
        if UPLO == 'U':     # Reflect to account for symmetry.
            return anp.triu(dx) + anp.tril(dx, -1).T
        else:
            return anp.tril(dx) + anp.triu(dx, 1).T
    return eigh_grad
eigh.defgrad(make_grad_eigh)



def make_grad_cholesky3(ans, x):
    """Computes reverse mode derivative of the Cholesky decomposition of A.
    Source: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf"""
    def gradient_product(g):
        L = ans                 #make new names to match Giles' paper
        A = x
        Lbar = g
        Abar = anp.zeros(A.shape)                 #initialize the return variable
        n = A.shape[0]
        for i in xrange(n-1,-1,-1):        # Perform the Cholesky decomposition
            for j in xrange(i,-1,-1):
                if (j == i):
                    Abar[i][i] = 0.5 * Lbar[i][i]/L[i][i]
                else:
                    Abar[i][j] = Lbar[i][j]/L[j][j]
                    Lbar[j][j] = Lbar[j][j] - Lbar[i][j]*L[i][j]/L[j][j]
                for k in xrange(j-1,-1,-1):
                    Lbar[i][k] = Lbar[i][k] - Abar[i][j]*L[j][k]
                    Lbar[j][k] = Lbar[j][k] - Abar[i][j]*L[i][k]
        return Abar
    return gradient_product


def make_grad_cholesky4(ans, x):
    """Computes reverse mode derivative of the Cholesky decomposition of A.
    Source: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf"""
    def gradient_product(g):
        g = anp.copy(g)
        Abar = anp.zeros(x.shape)                 #initialize the return variable
        n = x.shape[0]
        for i in xrange(n - 1, -1, -1):
            for j in xrange(i, -1, -1):
                Aij = g[i, j] / ans[j, j]
                if j == i:
                    Aij *= 0.5
                g[i, :] -= Aij * ans[j, :]
                g[j, :] -= Aij * ans[i, :]
                Abar[i, j] = Aij
        return Abar
    return gradient_product


def lower_half_symm(mat):
    # Takes the lower half of the matrix, and half the diagonal.
    # Necessary since cholesky only uses lower half of covariance matrix.
    return 0.5 * (anp.tril(mat) + anp.triu(mat, 1).T)

def lower_half_symm_nohalf(mat):
    # Takes the lower half of the matrix, and half the diagonal.
    # Necessary since cholesky only uses lower half of covariance matrix.
    return anp.tril(mat) + anp.triu(mat).T

def lower_half_take(mat):
    # Takes the lower half of the matrix, and half the diagonal.
    # Necessary since cholesky only uses lower half of covariance matrix.
    return 0.5 * anp.diag(anp.diag(mat)) + anp.tril(mat, -1)

def make_grad_cholesky(ans, x):
    # Things I've figured out:
    # 1. We'll need to not use the upper half of g.
    def gradient_product(g):
        outgrad = anp.tril(g)
        return lower_half_symm_nohalf(dot(outgrad, inv(ans)))
     return gradient_product
cholesky.defgrad(make_grad_cholesky)
