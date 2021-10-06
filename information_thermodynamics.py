########################################################################
#
#  Implement quantities defined in information thermodynamics theory.
#  06/10/2021: first commit
#
# References:
# - 非平衡統計力学の基礎理論
#   - http:/http://sosuke110.com/NoteBenkyokai.pdf
#   - For implementation.
# - Calculating Mutual Information in Python
#   - https://www.roelpeters.be/calculating-mutual-information-in-python/
#   - For test of implemented functions and libraries.
#   - ***Note: This article employs np.log2 instead of np.log***
#
########################################################################

import numpy as np
from scipy import stats


def check_probability_sum(p):
    """
    Check probability distribution.
    Input:
        1D ndarray $p(x)$ or
        2D ndarray $p(x,y)$
    Output:
        True if $\sum_x p(x) = 1$ or $\sum_{x,y} p(x,y) = 1$, False if not.
    """

    if p.ndim == 1:
        return np.isclose(sum(p[i] for i in range(len(p))), 1.0, rtol=1e-5)
    elif p.ndim == 2:
        X_range = np.shape(p)[0]
        Y_range = np.shape(p)[1]
        return np.isclose(
            sum(sum(p[i][j] for i in range(X_range)) for j in range(Y_range)),
            1.0, rtol=1e-5
        )


def shannon_entropy(p):
    """
    Compute Shannon entropy.
    Input:
        1D ndarray $p(x)$
    Output:
        Float $S(X) = - \sum_x p(x) \log p(x)$
    """

    if p.ndim != 1:
        raise ValueError('Input array p must be 1D ndarray')

    return sum(-p[i] * np.log(p[i]) for i in range(len(p)))


def relative_entropy(p, q):
    """
    Compute the relative entropy
    or the Kullback-Leibler distance.
    Input:
        1D ndarray * 2 $p(x), q(x)$
    Output:
        Float $D_\mathrm{KL}(p(X) || q(X)) = \sum_x p(x) \log \frac{p(x)}{q(x)}$
    """

    if p.ndim != 1:
        raise ValueError('Input array p must be 1D ndarray')
    if q.ndim != 1:
        raise ValueError('Input array q must be 1D ndarray')

    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))


# main
if __name__ == '__main__':

    # Entropy
    x = np.array([0.95, 0.05])
    print("Entropy of x=", x)
    print("Satisfy sum(x)=1:", check_probability_sum(x))
    print("Code: ", '{:6.4f}'.format(
        shannon_entropy(x)
    ))
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(x, base=np.e)
    ))
    print("")

    # Relative entropy (Kullback-Leibler divergence)
    y = np.array([0.2, 0.8])
    print("Relative entropy of x=", x, "and y=", y)
    print("Satisfy sum(y)=1:", check_probability_sum(y))
    print("Code: ", '{:6.4f}'.format(
        relative_entropy(x, y)
    ))
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(pk=x, qk=y, base=np.e)
    ))
    print("")
