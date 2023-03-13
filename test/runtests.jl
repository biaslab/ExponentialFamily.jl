using ExponentialFamily
using Test

@testset "ExponentialFamily" begin
    include("test_distributions.jl")
    include("test_natural_parameters.jl")
    include("distributions/test_common.jl")
    include("distributions/test_bernoulli.jl")
    include("distributions/test_beta.jl")
    include("distributions/test_categorical.jl")
    include("distributions/test_contingency.jl")
    include("distributions/test_dirichlet_matrix.jl")
    include("distributions/test_dirichlet.jl")
    include("distributions/test_exponential.jl")
    include("distributions/test_gamma.jl")
    include("distributions/test_gamma_inverse.jl")
    include("distributions/test_mv_normal_mean_covariance.jl")
    include("distributions/test_mv_normal_mean_precision.jl")
    include("distributions/test_mv_normal_weighted_mean_precision.jl")
    include("distributions/test_normal_mean_variance.jl")
    include("distributions/test_normal_mean_precision.jl")
    include("distributions/test_normal_weighted_mean_precision.jl")
    include("distributions/test_normal.jl")
    include("distributions/test_wishart.jl")
    include("distributions/test_wishart_inverse.jl")
    include("distributions/test_erlang.jl")
    include("distributions/test_von_mises_fisher.jl")
end
