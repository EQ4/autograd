from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad


# TODO write a monad for tracking likelihoods and maybe parameter evolution?

### util

def fixed_point(f, x0):
    x1 = f(x0)
    while different(x0, x1):
        x0, x1 = x1, f(x1)
    return x1


def different(tup1, tup2):
    return not all(map(np.allclose, tup1, tup2))


def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)
    return a / replace_zeros(a.sum(-1, keepdims=True))


def inner(a, b):
    return np.sum(a * b)


### Gaussian exp fam

def gaussian():
    def eta(theta):
        mu, Sigma = theta
        J = np.linalg.inv(Sigma)
        h = np.dot(J, mu)
        return np.vstack((-1./2*J, h))

    def statistic(y):
        return np.vstack((np.outer(y,y), y))

    def logZ(eta):
        J, h = -2*eta[:-1], eta[-1]
        return -1./2 * np.dot(h, np.linalg.solve(J, h)) \
            + 1./2 * np.log(np.linalg.det(J))

    def max_likelihood(expected_stats, n):
        yyT, y = expected_stats[:-1], expected_stats[-1]
        mu = y / n
        Sigma = yyT / n - np.outer(mu, mu)
        return mu, Sigma

    return eta, statistic, logZ, max_likelihood


### exp fam HMM EM

def EM(init_params, data, expfam_fns):
    eta, statistic, logZ, max_likelihood = expfam_fns

    def EM_update(params):
        return M_step(E_step(params, data))

    def E_step(params, data):
        pi, A, thetas = params
        natural_params = np.log(pi), np.log(A), map(eta, thetas), \
            map(lambda theta: -logZ(eta(theta)), thetas)
        import ipdb; ipdb.set_trace()
        return grad(hmm_log_partition_function)(natural_params, data)

    def M_step(expected_stats):
        E_init, E_trans, E_obs_statistics, E_ns = expected_stats
        pi, A = normalize(E_init), normalize(E_trans)
        thetas = map(max_likelihood, E_obs_statistics, E_ns)
        return pi, A, thetas

    def hmm_log_partition_function(natural_params, data):
        log_pi, log_A, etas, neglogZs = natural_params
        log_alpha = log_pi
        for y in data:
            log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) \
                + log_likelihoods(y, etas, neglogZs)
        return logsumexp(log_alpha)

    def log_likelihoods(y, etas, neglogZs):
        def log_likelihood(eta, neglogZ):
            return inner(eta, statistic(y)) + neglogZ
        return np.array(map(log_likelihood, etas, neglogZs))

    return fixed_point(EM_update, init_params)


if __name__ == '__main__':
    np.random.seed(0)
    np.seterr(divide='ignore', invalid='raise')

    data = npr.randn(10,2)  # TODO

    N = 2
    D = data.shape[1]

    def rand_gaussian(D):
        return npr.randn(D), np.eye(D)

    init_pi = normalize(npr.rand(N))
    init_A = normalize(npr.rand(N, N))
    init_obs_params = [rand_gaussian(D) for _ in range(N)]
    init_params = (init_pi, init_A, init_obs_params)

    pi, A, thetas = EM(init_params, data, gaussian())
