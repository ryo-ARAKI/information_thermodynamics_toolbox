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
export check_probability_sum, marginal_probability
export shannon_entropy, relative_entropy

"""
Check probability distribution.
Input:
    1D array p(x) or
    2D array p(x,y)
Output:
    True if sum_x p(x) = 1 or sum_{x,y} p(x,y) = 1, False if not.
"""
function check_probability_sum(p)
    return isapprox(sum(p), 1.0, rtol = 1e-5)
end

"""
Compute marginal probability.
Input:
    2D array p(x,y)
Output:
    1D array p(x) = sum_y p(x,y)
"""
function marginal_probability(p)

    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    return vec(sum(p, dims = 2))
end

"""
Compute Shannon entropy.
Input:
    1D or 2D array p(x)
Output:
    1D: Float S(X) = - sum_x p(x) log p(x)
    2D: Float S(X, Y) = - sum_{x,y} p(x,y) log p(x,y)
"""
function shannon_entropy(p)
    return sum(-p .* log.(p))
end

"""
Compute the relative entropy
or the Kullback-Leibler distance.
Input:
    1D array * 2 p(x), q(x)
Output:
    Float
    D_mathrm{KL}(p(x) || q(x))
        = sum_x p(x) log frac{p(x)}{q(x)},
"""
function relative_entropy(p, q)
    if length(size(p)) != length(size(q))
        throw(error("Input array p and q must have same dimension"))
    end
    if length(size(p)) != 1
        throw(error("Input array p must be 1D array"))
    end

    return sum(p .* log.(p ./ q))
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
    println(@sprintf "Code: %6.4f\n" shannon_entropy(px))

    println("Joint entropy of p(x,y)")
    println("Library: NOT IMPLEMENTED")
    println(@sprintf "Code: %6.4f\n" shannon_entropy(H.weights))

    # Relative entropy (Kullback-Leibler divergence)
    py = marginal_probability(transpose(H.weights))
    println("Relative entropy of p(x)=", px, "and q(x)=", py)
    println("Satisfy sum(q)=1:", check_probability_sum(py))
    println(px)
    println(py)
    println(@sprintf "Code: %6.4f\n" relative_entropy(px, py))
end

main()
