module ExponentialFamily

using BayesBase
using TinyHugeNumbers
using FastCholesky

using IrrationalConstants:
    twoπ,       # 2π
    fourπ,      # 4π
    halfπ,      # π / 2
    quartπ,     # π / 4
    invπ,       # 1 / π
    twoinvπ,    # 2 / π
    fourinvπ,   # 4 / π
    inv2π,      # 1 / (2π)
    inv4π,      # 1 / (4π)
    sqrt2,      # √2
    sqrt3,      # √3
    sqrtπ,      # √π
    sqrt2π,     # √2π
    sqrt4π,     # √4π
    sqrthalfπ,  # √(π / 2)
    invsqrt2,   # 1 / √2
    invsqrt2π,  # 1 / √2π
    loghalf,    # log(1 / 2)
    logtwo,     # log(2)
    logπ,       # log(π)
    log2π,      # log(2π)
    log4π       # log(4π)

import Base: show, maximum, minimum
import Base: convert, promote_rule

include("common.jl")
include("exponential_family.jl")

include("distributions/bernoulli.jl")
include("distributions/categorical.jl")
include("distributions/gamma_family/gamma_shape_rate.jl")
include("distributions/gamma_family/gamma_shape_scale.jl")
include("distributions/gamma_family/gamma_family.jl")
include("distributions/normal_family/normal_mean_variance.jl")
include("distributions/normal_family/normal_mean_precision.jl")
include("distributions/normal_family/normal_weighted_mean_precision.jl")
include("distributions/normal_family/mv_normal_mean_covariance.jl")
include("distributions/normal_family/mv_normal_mean_precision.jl")
include("distributions/normal_family/mv_normal_weighted_mean_precision.jl")
include("distributions/normal_family/mv_normal_mean_scale_precision.jl")
include("distributions/normal_family/normal_family.jl")
include("distributions/gamma_inverse.jl")
include("distributions/geometric.jl")
include("distributions/matrix_dirichlet.jl")
include("distributions/dirichlet.jl")
include("distributions/beta.jl")
include("distributions/lognormal.jl")
include("distributions/binomial.jl")
# include("distributions/multinomial.jl")
include("distributions/wishart.jl")
include("distributions/wishart_inverse.jl")
# include("distributions/contingency.jl")
include("distributions/erlang.jl")
include("distributions/exponential.jl")
include("distributions/von_mises_fisher.jl")
include("distributions/von_mises.jl")
include("distributions/pareto.jl")
# include("distributions/continuous_bernoulli.jl")
include("distributions/negative_binomial.jl")
include("distributions/rayleigh.jl")
include("distributions/weibull.jl")
include("distributions/laplace.jl")
include("distributions/poisson.jl")
include("distributions/chi_squared.jl")
include("distributions/mv_normal_wishart.jl")
include("distributions/normal_gamma.jl")
include("distributions/tensor_dirichlet.jl")

end
