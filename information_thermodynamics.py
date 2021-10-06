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
from sklearn.metrics import mutual_info_score


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


def marginal_probability(p):
    """
    Compute marginal probability.
    Input:
        2D ndarray $p(x,y)$
    Output:
        1D ndarray $p(x) = \sum_y p(x,y)$
    """

    if p.ndim != 2:
        raise ValueError('Input array p must be 2D ndarray')

    Y_range = np.shape(p)[1]

    return sum(p[:][j] for j in range(Y_range))


def conditional_probability(p):
    """
    Compute conditional probability distribution.
    Input:
        2D ndarray $p(x,y)$
    Output:
        2D ndarray $p(x|y) = p(x,y) / p(y)$
    """

    if p.ndim != 2:
        raise ValueError('Input array p must be 2D ndarray')

    return p / marginal_probability(p)


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


def conditional_entropy(p):
    """
    Compute conditional entropy.
    Input:
        2D ndarray $p(x,y)$
    Output:
        Float $S(X|Y) = - \sum_{x,y} p(x,y) \log p(x|y)$
    """

    if p.ndim != 2:
        raise ValueError('Input array p must be 2D ndarray')

    X_range = np.shape(p)[0]
    Y_range = np.shape(p)[1]

    # Conditional probability distribution p(x|y)
    px_cond_y = conditional_probability(p)

    return sum(
        sum(-p[i][j] * np.log(px_cond_y[i][j]) for i in range(X_range))
        for j in range(Y_range)
    )


def relative_entropy(p, q):
    """
    Compute the relative entropy
    or the Kullback-Leibler distance.
    Input:
        1D ndarray * 2 $p(x), q(x)$ or
    Output:
        Float
        $$
        D_\mathrm{KL}(p(x) || q(x))
            = \sum_x p(x) \log \frac{p(x)}{q(x)},
        $$
    """

    if p.ndim != q.ndim:
        raise ValueError('Input array p and q must have same dimension')
    if p.ndim != 1:
        raise ValueError('Input array p must be 1D ndarray')

    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))


def joint_relative_entropy(p, q):
    """
    Compute the joint relative entropy.
    Input:
        2D ndarray * 2 $p(x,y), q(x,y)$
    Output:
        Float
        $$
        D_\mathrm{KL}(p(X,Y) || q(X,Y))
            = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{q(x,y)}
        $$
    """

    if p.ndim != q.ndim:
        raise ValueError('Input array p and q must have same dimension')
    if p.ndim != 2:
        raise ValueError('Input array p must be 2D ndarray')

    X_range = np.shape(p)[0]
    Y_range = np.shape(p)[1]

    return sum(
        sum(p[i][j] * np.log(p[i][j]/q[i][j]) for i in range(X_range))
        for j in range(Y_range)
    )


def mutual_information(p):
    """
    Compute the mutual information.
    Input:
        2D ndarray $p(x,y)$
    Output:
        Float
        $$
        I(X;Y)
            = \sum_x p(x) [log p(x,y) - log p(x) - log p(y)]
            = S(X) - S(X|Y)
            = D_\mathrm{KL} (p(x,y) || p(x)p(y))
        $$
    """

    if p.ndim != 2:
        raise ValueError('Input array p must be 2D ndarray')

    X_range = np.shape(p)[0]
    Y_range = np.shape(p)[1]

    # Conditional probability distribution p(x|y) and q(x|y)
    px = marginal_probability(p)
    py = marginal_probability(p.transpose()).transpose()

    # I(X;Y) = \sum_x p(x) [log p(x,y) - log p(x) - log p(y)]
    MI = sum(sum(
        p[i][j] * (np.log(p[i][j]) - np.log(px[i]) - np.log(py[j]))
        for i in range(X_range))
        for j in range(Y_range)
    )
    # I(X;Y) = S(X) - S(X|Y)
    MI_se = shannon_entropy(marginal_probability(p)) - conditional_entropy(p)
    # I(X;Y) = D_\mathrm{KL} (p(x,y) || p(x)p(y))
    MI_kl = joint_relative_entropy(p, np.tensordot(px, py, axes=0))

    print("\n=====")
    print("Compute MI in different ways")
    print("By definition     =", '{:6.4}'.format(MI))
    print("By Shannon entropy=", '{:6.4}'.format(MI_se))
    print("By KL divergence  =", '{:6.4}'.format(MI_kl))
    print("=====\n")

    return MI


# main
if __name__ == '__main__':

    # Entropy
    x = np.array([0.95, 0.05])
    print("Entropy of p(x)=", x)
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
    print("Relative entropy of p(x)=", x, "and q(x)=", y)
    print("Satisfy sum(q)=1:", check_probability_sum(y))
    print("Code: ", '{:6.4f}'.format(
        relative_entropy(x, y)
    ))
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(pk=x, qk=y, base=np.e)
    ))
    print("")

    # Prepare 2D probability distribution p(x,y) by 2D histgram
    # Set edge of bin
    xedges = [0.0, 0.5, 1.0]
    yedges = [0.0, 0.5, 1.0]
    # Set X and Y (binary array)
    x = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    # Prepare histgram = p(x,y)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H /= np.sum(H)
    # Substitute 0 by finite value to avoid log(0) error
    H[H == 0] = float(1e-8)

    # Mutual information
    print("Mutual information of p(x,y):\n", H)
    print("Satisfy sum(p)=1:", check_probability_sum(H))
    print("sklearn: ", '{:6.4f}'.format(
        mutual_info_score(x, y)
    ))
    print("Code:    ", '{:6.4f}'.format(
        mutual_information(H)
    ))
    print("")
