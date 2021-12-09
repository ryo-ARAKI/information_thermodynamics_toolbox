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
using Printf
export check_probability_sum, marginal_probability
export shannon_entropy, relative_entropy, conditional_relative_entropy
export mutual_information

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
Compute conditional probability distribution.
Input:
    2D array p(x,y)
Output:
    2D array p(x|y) = p(x,y) / p(y)
"""
function conditional_probability(p)
    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    py = marginal_probability(transpose(p))
    px_1 = vec(ones(length(py), 1))

    return p ./ (px_1 .* transpose(py))
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
Compute conditional entropy.
Input:
    2D darray p(x,y)
Output:
    Float S(X|Y) = - sum_{x,y} p(x,y) log p(x|y)
"""
function conditional_entropy(p)
    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    # Conditional probability distribution p(x|y)
    px_cond_y = conditional_probability(p)

    return sum(-p .* log.(px_cond_y))
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

"""
Compute the joint relative entropy.
Input:
    2D array * 2 p(x,y), q(x,y)
Output:
    Float
    D_mathrm{KL}(p(X,Y) || q(X,Y))
        = sum_{x,y} p(x,y) log frac{p(x,y)}{q(x,y)}
"""
function joint_relative_entropy(p, q)
    if length(size(p)) != length(size(q))
        throw(error("Input array p and q must have same dimension"))
    end
    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    return sum(p .* log.(p ./ q))
end

"""
Compute the conditional relative entropy.
Input:
    2D array * 2 p(x,y), q(x,y)
Output:
    Float D_mathrm{KL}(p(X|Y) || q(X|Y)) = sum_{x,y} p(x,y) log frac{p(x|y)}{q(x|y)}
"""
function conditional_relative_entropy(p, q)
    if length(size(p)) != length(size(q))
        throw(error("Input array p and q must have same dimension"))
    end
    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    # Conditional probability distribution p(x|y) and q(x|y)
    px_cond_y = conditional_probability(p)
    qx_cond_y = conditional_probability(q)

    return sum(p .* log.(px_cond_y ./ qx_cond_y))
end

"""
Compute the mutual information.
Input:
    2D array p(x,y)
Output:
    Float
    I(X;Y)
        = sum_x p(x) [log p(x,y) - log p(x) - log p(y)]
        = S(X) - S(X|Y)
        = D_mathrm{KL} (p(x,y) || p(x)p(y))
"""
function mutual_information(p)
    if length(size(p)) != 2
        throw(error("Input array p must be 2D array"))
    end

    # Marginal probability distribution p(x) and q(y)
    px = marginal_probability(p)
    py = marginal_probability(transpose(p))

    # I(X;Y) = \sum_x \sum_y p(x) [log p(x,y) - log p(x) - log p(y)]
    MI = 0.0
    for row = 1:size(p, 1), col = 1:size(p, 2)
        MI += p[row, col] * (
            log(p[row, col]) - log(px[row]) - log(py[col])
        )
    end
    # I(X;Y) = S(X) - S(X|Y)
    MI_se = shannon_entropy(px) - conditional_entropy(p)
    # I(X;Y) = D_\mathrm{KL} (p(x,y) || p(x)p(y))
    MI_kl = joint_relative_entropy(p, px .* transpose(py))

    println("\n=====")
    println("Compute MI in different ways")
    println(@sprintf "By definition     = %6.4f" MI)
    println(@sprintf "By Shannon entropy= %6.4f" MI_se)
    println(@sprintf "By KL divergence  = %6.4f" MI_kl)
    println("=====\n")

    return MI
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
    println("2D probability distribution p(x,y)\n", histogram.weights, "\n")

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
    println("Satisfy sum(x)=1: ", check_probability_sum(px))
    println(@sprintf "Code: %6.4f\n" shannon_entropy(px))

    println("Joint entropy of p(x,y)")
    println("Library: NOT IMPLEMENTED")
    println(@sprintf "Code: %6.4f\n" shannon_entropy(H.weights))

    # Relative entropy (Kullback-Leibler divergence)
    py = marginal_probability(transpose(H.weights))
    println("Relative entropy of p(x)=", px, "and q(x)=", py)
    println("Satisfy sum(q)=1: ", check_probability_sum(py))
    println(@sprintf "Code: %6.4f\n" relative_entropy(px, py))

    println("Conditional relative entropy of p(x,y) & p(x,y)")
    println("Library: NOT IMPLEMENTED")
    println(@sprintf "Code: %6.4f\n" conditional_relative_entropy(H.weights, H.weights))

    # Mutual information
    println("Mutual information of p(x,y)")
    println("Satisfy sum(p)=1: ", check_probability_sum(H.weights))
    println(@sprintf "Code: %6.4f\n" mutual_information(H.weights))
end

main()
