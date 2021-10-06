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


# main
if __name__ == '__main__':

    # Entropy
    x = [0.95, 0.05]
    print("Entropy of x=", x)
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(x, base=np.e)
    ))
    print("")

    # Relative entropy (Kullback-Leibler divergence)
    y = [0.2, 0.8]
    print("Relative entropy of x=", x, "and y=", y)
    print("Scipy:", '{:6.4f}'.format(
        stats.entropy(pk=x, qk=y, base=np.e)
    ))
    print("")
