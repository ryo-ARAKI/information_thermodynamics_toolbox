#=
Implement quantities defined in information thermodynamics theory.
09/12/2021: first commit

References:
    - 非平衡統計力学の基礎理論
        - http:/http://sosuke110.com/NoteBenkyokai.pdf
    - For implementation.
        - Calculating Mutual Information in Python
    - https://www.roelpeters.be/calculating-mutual-information-in-python/
    - For test of implemented functions and libraries.
        - ***Note: This article employs np.log2 instead of np.log***
=#

module InformationThermodynamics
export check_probability_sum
export marginal_probability

"""
Check probability distribution.
Input:
    1D ndarray p(x) or
    2D ndarray p(x,y)
Output:
    True if sum_x p(x) = 1 or sum_{x,y} p(x,y) = 1, False if not.
"""
function check_probability_sum(p)
    return isapprox(sum(p), 1.0, rtol = 1e-5)
end

"""
Compute marginal probability.
Input:
    2D ndarray p(x,y)
Output:
    1D ndarray p(x) = sum_y p(x,y)
"""
function marginal_probability(p)

    if length(size(p)) != 2
        throw(error("Input array p must be 2D ndarray"))
    end

    return sum(p, dims = 2)

end

end


"""
Prepare 2D probability distribution p(x,y) by 2D histogram
"""
function prepare_2d_probability_distribution()

    # Set edge of bin
    eps = 1e-8
    xedges = [-0.0-eps, 0.5, 1.0+eps]
    yedges = [-0.0-eps, 0.5, 1.0+eps]
    # Set X and Y (binary array)
    x = [1, 1, 1, 0, 0, 1, 0, 0, 0, 1]
    y = [1, 1, 1, 0, 0, 1, 0, 0, 0, 1]
    # Prepare histogram = p(x,y)
    histogram = fit(Histogram{Float64}, (x, y), (xedges, yedges))
    histogram = normalize(histogram, mode = :probability)
    # Substitute 0 by finite value to avoid log(0) error
    histogram.weights[histogram.weights.==0.0] .= 1.0e-8

    # Stdout probability distributions
    print("2D probability distribution p(x,y)\n", histogram.weights, "\n")

    return x, y, histogram
end


# main
using Printf
using StatsBase
using LinearAlgebra
using .InformationThermodynamics
function main()

    # Prepare test data
    x, y, H = prepare_2d_probability_distribution()

    # Entropy
    px = marginal_probability(H.weights)
    println("Entropy of p(x)=", px)
    println("Satisfy sum(x)=1:", check_probability_sum(px))
end

main()
