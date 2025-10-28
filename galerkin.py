import numpy as np
import scipy.sparse as sparse
import sympy as sp
from numpy.polynomial import Chebyshev as Cheb
from numpy.polynomial import Legendre as Leg
from scipy.integrate import quad

x = sp.Symbol("x")


def map_reference_domain(x, d, r):
    return r[0] + (r[1] - r[0]) * (x - d[0]) / (d[1] - d[0])


def map_true_domain(x, d, r):
    return d[0] + (d[1] - d[0]) * (x - r[0]) / (r[1] - r[0])


def map_expression_true_domain(u, x, d, r):
    if d != r:
        u = sp.sympify(u)
        xm = map_true_domain(x, d, r)
        u = u.replace(x, xm)
    return u


class FunctionSpace:
    def __init__(self, N, domain=(-1, 1)):
        self.N = N
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    @property
    def reference_domain(self):
        raise RuntimeError

    @property
    def domain_factor(self):
        d = self.domain
        r = self.reference_domain
        return (d[1] - d[0]) / (r[1] - r[0])

    def mesh(self, N=None):
        d = self.domain
        n = N if N is not None else self.N
        return np.linspace(d[0], d[1], n + 1)

    def weight(self, x=x):
        return 1

    def basis_function(self, j, sympy=False):
        raise RuntimeError

    def derivative_basis_function(self, j, k=1):
        raise RuntimeError

    def evaluate_basis_function(self, Xj, j):
        return self.basis_function(j)(Xj)

    def evaluate_derivative_basis_function(self, Xj, j, k=1):
        return self.derivative_basis_function(j, k=k)(Xj)

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh

    def eval_basis_function_all(self, Xj):
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_basis_function(Xj, j)
        return P

    def eval_derivative_basis_function_all(self, Xj, k=1):
        Xj = np.atleast_1d(Xj)
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_derivative_basis_function(Xj, j, k=k)
        return P

    def inner_product(self, u):
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = self.domain_factor
        r = self.reference_domain
        for i in range(self.N + 1):
            psi = self.basis_function(i)

            def uv(Xj):
                return us(Xj) * psi(Xj)

            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0]
        return uj

    def mass_matrix(self):
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self))


class Legendre(FunctionSpace):
    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain)

    @property
    def reference_domain(self):
        return (-1, 1)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.legendre(j, x)
        return Leg.basis(j)

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k)

    def L2_norm_sq(self, N):
        n = np.arange(N, dtype=float)
        return 2.0 / (2.0 * n + 1.0)

    def mass_matrix(self):
        diag = self.L2_norm_sq(self.N + 1)
        return sparse.diags([diag], [0], shape=(self.N + 1, self.N + 1), format="csr")

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.legendre.legval(Xj, uh)


class Chebyshev(FunctionSpace):
    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain)

    @property
    def reference_domain(self):
        return (-1, 1)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x))
        return Cheb.basis(j)

    def derivative_basis_function(self, j, k=1):
        return Cheb.basis(j).deriv(k)

    def weight(self, x=x):
        return 1 / sp.sqrt(1 - x**2)

    def L2_norm_sq(self, N):
        out = np.full(N, 0.5 * np.pi, dtype=float)
        if N > 0:
            out[0] = np.pi
        return out

    def mass_matrix(self):
        diag = self.L2_norm_sq(self.N + 1)
        return sparse.diags([diag], [0], shape=(self.N + 1, self.N + 1), format="csr")

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u):
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        # change of variables to x=cos(theta)
        us = sp.simplify(us.subs(x, sp.cos(x)), inverse=True)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = float(self.domain_factor)
        k = sp.Symbol("k")
        basis = sp.lambdify(
            (k, x),
            sp.simplify(self.basis_function(k, True).subs(x, sp.cos(x), inverse=True)),
        )
        for i in range(self.N + 1):

            def uv(Xj, j):
                return us(Xj) * basis(j, Xj)

            uj[i] = float(h) * quad(uv, 0, np.pi, args=(i,))[0]
        return uj


class Trigonometric(FunctionSpace):
    """Base class for trigonometric function spaces"""

    @property
    def reference_domain(self):
        return (0, 1)

    def mass_matrix(self):
        return sparse.diags(
            [self.L2_norm_sq(self.N + 1)], [0], (self.N + 1, self.N + 1), format="csr"
        )

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)


class Sines(Trigonometric):
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.sin((j + 1) * sp.pi * x)
        return lambda Xj: np.sin((j + 1) * np.pi * Xj)

    def derivative_basis_function(self, j, k=1):
        scale = ((j + 1) * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.sin((j + 1) * np.pi * Xj)
        else:
            return lambda Xj: scale * np.cos((j + 1) * np.pi * Xj)

    def L2_norm_sq(self, N):
        return np.full(N, 0.5, dtype=float)


class Cosines(Trigonometric):
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.pi * x)
        return lambda Xj: np.cos(j * np.pi * Xj)

    def derivative_basis_function(self, j, k=1):
        factor = (j * np.pi) ** k
        if k % 2 == 0:
            return lambda Xj: factor * np.cos(j * np.pi * Xj)
        else:
            return lambda Xj: -factor * np.sin(j * np.pi * Xj)

    def L2_norm_sq(self, N):
        out = np.full(N, 0.5, dtype=float)
        if N > 0:
            out[0] = 1.0
        return out


# Create classes to hold the boundary function


class Dirichlet:
    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = (
            bc[0] * (d[1] - x) / h + bc[1] * (x - d[0]) / h
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)


class Neumann:
    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = bc[0] / h * (d[1] * x - x**2 / 2) + bc[1] / h * (
            x**2 / 2 - d[0] * x
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)


class Composite(FunctionSpace):
    r"""Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.
    """

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self):
        M = sparse.diags(
            [self.L2_norm_sq(self.N + 3)],
            [0],
            shape=(self.N + 3, self.N + 3),
            format="csr",
        )
        return self.S @ M @ self.S.T

    # Critical: derivatives of composite basis ψ_j (not the orthogonal Q_j)
    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k)


class DirichletLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.legendre(j, x) - sp.legendre(j + 2, x)
        return Leg.basis(j) - Leg.basis(j + 2)


class NeumannLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        Legendre.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)
        rows, cols, data = [], [], []
        for i in range(N + 1):
            gamma = (i * (i + 1)) / ((i + 2) * (i + 3))  # cancels derivative at ±1
            rows += [i, i]
            cols += [i, i + 2]
            data += [1.0, -gamma]
        self.S = sparse.csr_matrix((data, (rows, cols)), shape=(N + 1, N + 3))

    def basis_function(self, j, sympy=False):
        if sympy:
            gamma = (j * (j + 1)) / ((j + 2) * (j + 3))
            return sp.legendre(j, x) - gamma * sp.legendre(j + 2, x)
        return Leg.basis(j) - ((j * (j + 1)) / ((j + 2) * (j + 3))) * Leg.basis(j + 2)


class DirichletChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x)) - sp.cos((j + 2) * sp.acos(x))
        return Cheb.basis(j) - Cheb.basis(j + 2)


class NeumannChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)
        rows, cols, data = [], [], []
        for i in range(N + 1):
            gamma = (i / (i + 2)) ** 2 if (i + 2) != 0 else 0.0
            rows += [i, i]
            cols += [i, i + 2]
            data += [1.0, -gamma]
        self.S = sparse.csr_matrix((data, (rows, cols)), shape=(N + 1, N + 3))

    def basis_function(self, j, sympy=False):
        gamma = (j / (j + 2)) ** 
