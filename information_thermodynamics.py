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


class InformationThermodynamics():

    def __init__(self, H):
        self.xrange = np.shape(H)[0]
        self.yrange = np.shape(H)[1]


    def check_probability_sum(self, p):
        """
        Check probability distribution.
        Input:
            1D ndarray $p(x)$ or
            2D ndarray $p(x,y)$
        Output:
            True if $\sum_x p(x) = 1$ or $\sum_{x,y} p(x,y) = 1$, False if not.
        """

        if p.ndim == 1:
            return np.isclose(sum(p[i] for i in range(self.xrange)), 1.0, rtol=1e-5)
        elif p.ndim == 2:
            return np.isclose(
                sum(sum(p[i][j] for i in range(self.xrange)) for j in range(self.yrange)),
                1.0, rtol=1e-5
            )


    def marginal_probability(self, p):
        """
        Compute marginal probability.
        Input:
            2D ndarray $p(x,y)$
        Output:
            1D ndarray $p(x) = \sum_y p(x,y)$
        """

        if p.ndim != 2:
            raise ValueError('Input array p must be 2D ndarray')

        return sum(p[:][j] for j in range(self.yrange))


    def conditional_probability(self, p):
        """
        Compute conditional probability distribution.
        Input:
            2D ndarray $p(x,y)$
        Output:
            2D ndarray $p(x|y) = p(x,y) / p(y)$
        """

        if p.ndim != 2:
            raise ValueError('Input array p must be 2D ndarray')

        return p / self.marginal_probability(p)


    def shannon_entropy(self, p):
        """
        Compute Shannon entropy.
        Input:
            1D ndarray $p(x)$
        Output:
            Float $S(X) = - \sum_x p(x) \log p(x)$
        """

        if p.ndim != 1:
            raise ValueError('Input array p must be 1D ndarray')

        return sum(-p[i] * np.log(p[i]) for i in range(self.xrange))


    def joint_entropy(self, p):
        """
        Compute joint entropy.
        Input:
            2D ndarray $p(x,y)$
        Output:
            Float $S(X, Y) = - \sum_{x,y} p(x,y) \log p(x,y)$
        """

        if p.ndim != 2:
            raise ValueError('Input array p must be 2D ndarray')

        return sum(
            sum(-p[i][j] * np.log(p[i][j]) for i in range(self.xrange))
            for j in range(self.yrange)
        )


    def conditional_entropy(self, p):
        """
        Compute conditional entropy.
        Input:
            2D ndarray $p(x,y)$
        Output:
            Float $S(X|Y) = - \sum_{x,y} p(x,y) \log p(x|y)$
        """

        if p.ndim != 2:
            raise ValueError('Input array p must be 2D ndarray')

        # Conditional probability distribution p(x|y)
        px_cond_y = self.conditional_probability(p)

        return sum(
            sum(-p[i][j] * np.log(px_cond_y[i][j]) for i in range(self.xrange))
            for j in range(self.yrange)
        )


    def relative_entropy(self, p, q):
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

        return sum(p[i] * np.log(p[i]/q[i]) for i in range(self.xrange))


    def joint_relative_entropy(self, p, q):
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

        return sum(
            sum(p[i][j] * np.log(p[i][j]/q[i][j]) for i in range(self.xrange))
            for j in range(self.yrange)
        )


    def conditional_relative_entropy(self, p, q):
        """
        Compute the conditional relative entropy.
        Input:
            2D ndarray * 2 $p(x,y), q(x,y)$
        Output:
            Float $D_\mathrm{KL}(p(X|Y) || q(X|Y)) = \sum_{x,y} p(x,y) \log \frac{p(x|y)}{q(x|y)}$
        """

        if p.ndim != q.ndim:
            raise ValueError('Input array p and q must have same dimension')
        if p.ndim != 2:
            raise ValueError('Input array p must be 2D ndarray')

        # Conditional probability distribution p(x|y) and q(x|y)
        px_cond_y = self.conditional_probability(p)
        qx_cond_y = self.conditional_probability(q)

        return sum(sum(
            p[i][j] * np.log(px_cond_y[i]/qx_cond_y[i]) for i in range(self.xrange)
            ) for j in range(self.yrange)
        )


    def mutual_information(self, p):
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

        # Marginal probability distribution p(x) and q(y)
        px = self.marginal_probability(p)
        py = self.marginal_probability(p.transpose()).transpose()

        # I(X;Y) = \sum_x p(x) [log p(x,y) - log p(x) - log p(y)]
        MI = sum(sum(
            p[i][j] * (np.log(p[i][j]) - np.log(px[i]) - np.log(py[j]))
            for i in range(self.xrange))
            for j in range(self.yrange)
        )
        # I(X;Y) = S(X) - S(X|Y)
        MI_se = self.shannon_entropy(px) \
            - self.conditional_entropy(p)
        # I(X;Y) = D_\mathrm{KL} (p(x,y) || p(x)p(y))
        MI_kl = self.joint_relative_entropy(p, np.tensordot(px, py, axes=0))

        print("\n=====")
        print("Compute MI in different ways")
        print("By definition     =", '{:6.4}'.format(MI))
        print("By Shannon entropy=", '{:6.4}'.format(MI_se))
        print("By KL divergence  =", '{:6.4}'.format(MI_kl))
        print("=====\n")

        return MI


def prepare_2d_probability_distribution():
    """
    Prepare 2D probability distribution p(x,y) by 2D histogram
    """

    # Set edge of bin
    xedges = [0.0, 0.5, 1.0]
    yedges = [0.0, 0.5, 1.0]
    # Set X and Y (binary array)
    x = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    # Prepare histogram = p(x,y)
    histogram, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    histogram /= np.sum(histogram)
    # Substitute 0 by finite value to avoid log(0) error
    histogram[histogram == 0] = float(1e-8)

    # Stdout probability distributions
    print("2D probability distribution p(x,y)\n", histogram, "\n")

    return x, y, histogram


# main
if __name__ == '__main__':

    # Prepare test data
    x, y, H = prepare_2d_probability_distribution()


    # Declare class containing functions to compute
    # Information thermodynamics quantities
    InfoThermo = InformationThermodynamics(H)

    # Entropy
    px = InfoThermo.marginal_probability(H)
    print("Entropy of p(x)=", px)
    print("Satisfy sum(x)=1:", InfoThermo.check_probability_sum(px))
    print("Code: ", '{:6.4f}'.format(
        InfoThermo.shannon_entropy(px)
    ))
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(px, base=np.e)
    ), "\n")

    # Relative entropy (Kullback-Leibler divergence)
    py = InfoThermo.marginal_probability(H.transpose()).transpose()
    print("Relative entropy of p(x)=", px, "and q(x)=", py)
    print("Satisfy sum(q)=1:", InfoThermo.check_probability_sum(py))
    print("Code: ", '{:6.4f}'.format(
        InfoThermo.relative_entropy(px, py)
    ))
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(pk=px, qk=py, base=np.e)
    ), "\n")

    # Mutual information
    print("Mutual information of p(x,y)")
    print("Satisfy sum(p)=1:", InfoThermo.check_probability_sum(H))
    print("sklearn: ", '{:6.4f}'.format(
        mutual_info_score(x, y)
    ))
    print("Code:    ", '{:6.4f}'.format(
        InfoThermo.mutual_information(H)
    ), "\n")
