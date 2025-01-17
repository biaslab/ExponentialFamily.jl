export TensorDirichlet, ContinuousTensorDistribution

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf
using Distributions
using SpecialFunctions, LogExpFunctions

import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra, Random

using BlockArrays: BlockDiagonal

const ContinuousTensorDistribution = Distribution{ArrayLikeVariate, Continuous}

"""
    TensorDirichlet{T <: Real, N, A <: AbstractArray{T, N}, Ts} <: ContinuousTensorDistribution

A tensor-valued Dirichlet distribution, where `T` is the element type of the tensor `A`. The distribution generalizes the Dirichlet distribution to handle multiple sets of parameters organized in a tensor structure. This distribution collects multiple independent Dirichlet distributions into a single efficient interface. The Dirichlet counts for the independent Dirichlet distributions are stored along the first dimension of `a`. This distribution can be used as a conjugate prior to a Categorical distribution with mulitple switch cases (such as a discrete state-transition with controls).  

# Fields
- `a::A`: The tensor parameter of the distribution, where each slice represents parameters of a Dirichlet distribution
- `α0::Ts`: The sum of parameters along the first dimension.
- `lmnB::Ts`: The log multinomial beta function values for each slice.

The distribution models multiple independent Dirichlet distributions organized in a tensor structure, where each slice `a[:,i,j,...]` represents the parameters of an independent Dirichlet distribution.
"""
struct TensorDirichlet{T <: Real, N, A <: AbstractArray{T, N}, Ts} <: ContinuousTensorDistribution
    a::A
    α0::Ts
    lmnB::Ts
    function TensorDirichlet(alpha::AbstractArray{T, N}) where {T, N}
        if !all(x -> x > zero(x), alpha)
            throw(ArgumentError("All elements of the alpha tensor should be positive"))
        end
        alpha0 = sum(alpha; dims = 1)
        lmnB = sum(loggamma, alpha; dims = 1) - loggamma.(alpha0)
        new{T, N, typeof(alpha), typeof(alpha0)}(alpha, alpha0, lmnB)
    end
end

function BayesBase.logpdf(dist::TensorDirichlet{T, N, A, Ts}, xs::AbstractVector{A}) where {T, N, A, Ts}
    return map(x -> logpdf(dist, x), xs)
end

function BayesBase.pdf(dist::TensorDirichlet{R, N, A}, x::AbstractArray{T, N}) where {R, A, T <: Real, N}
    return exp(logpdf(dist, x))
end

function BayesBase.pdf(dist::TensorDirichlet, xs::AbstractVector)
    return map(x -> pdf(dist, x), xs)
end

BayesBase.params(dist::TensorDirichlet) = (dist.a,)

function unpack_parameters(::Type{TensorDirichlet}, packed, conditioner)
    packed = view(packed, 1:length(packed))
    return (reshape(packed, conditioner),)
end

function join_conditioner(::Type{TensorDirichlet}, cparams, _)
    return cparams
end

function separate_conditioner(::Type{TensorDirichlet}, tuple_of_θ)
    return (tuple_of_θ, size(tuple_of_θ[1]))
end

isbasemeasureconstant(::Type{TensorDirichlet}) = ConstantBaseMeasure()

getbasemeasure(::Type{TensorDirichlet}, conditioner) = (x) -> one(Float64)
getlogbasemeasure(::Type{TensorDirichlet}, conditioner) = (x) -> zero(Float64)

getsufficientstatistics(::Type{TensorDirichlet}, conditioner) = (x -> vmap(log, x),)

BayesBase.mean(dist::TensorDirichlet) = dist.a ./ dist.α0
BayesBase.mean(::BroadcastFunction{typeof(log)}, dist::TensorDirichlet) = digamma.(dist.a) .- digamma.(dist.α0)

function BayesBase.cov(dist::TensorDirichlet{T}) where {T}
    s = size(dist.a)
    news = (first(s), first(s), Base.tail(s)...)
    v = zeros(T, news)
    for i in CartesianIndices(Base.tail(size(dist.a)))
        v[:, :, i] .= cov(Dirichlet(dist.a[:, i]))
    end
    return v
end
function BayesBase.var(dist::TensorDirichlet{T, N, A, Ts}) where {T, N, A, Ts}
    α = dist.a
    α0 = dist.α0
    c = inv.(α0 .^ 2 .* (α0 .+ 1))
    v = α .* (α0 .- α) .* c
    return v
end
BayesBase.std(dist::TensorDirichlet) = sqrt.(var(dist))

Base.size(dist::TensorDirichlet) = size(dist.a)
Base.eltype(::TensorDirichlet{T}) where {T} = T

function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Int)
    return TensorDirichlet(ones(dims, dims))
end

function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Tuple)
    return TensorDirichlet(ones(Float64, dims))
end

function BayesBase.entropy(dist::TensorDirichlet)
    α = dist.a
    α0 = dist.α0
    lmnB = dist.lmnB
    return sum(-sum((α .- one(eltype(α))) .* (digamma.(α) .- digamma.(α0)); dims = 1) .+ lmnB)
end

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:TensorDirichlet}) = TensorDirichlet
BayesBase.promote_variate_type(::Type{ArrayLikeVariate}, ::Type{<:Dirichlet}) = TensorDirichlet

function BayesBase.rand(rng::AbstractRNG, dist::TensorDirichlet{T}) where {T}
    container = similar(dist.a)
    return rand!(rng, dist, container)
end

function BayesBase.rand(rng::AbstractRNG, dist::TensorDirichlet{T}, nsamples::Int64) where {T}
    container = [similar(dist.a) for _ in 1:nsamples]
    rand!(rng, dist, container)
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::TensorDirichlet, container::AbstractArray{T, N}) where {T <: Real, N}
    for (i, αi) in zip(eachindex(container), dist.a)
        @inbounds container[i] = rand(rng, Gamma(αi))
    end
    container .= container ./ sum(container; dims = 1)
end

# Add method for handling vector of arrays
function BayesBase.rand!(
    rng::AbstractRNG,
    dist::TensorDirichlet{T, N, A, Ts},
    container::AbstractArray{A, M}
) where {T <: Real, N, A <: AbstractArray{T, N}, Ts, M}
    for c in container
        size(c) == size(dist.a) || error("Size mismatch")
    end

    @inbounds for c in container
        rand!(rng, dist, c)
    end

    return container
end

function BayesBase.logpdf(dist::TensorDirichlet{R, N, A}, x::AbstractArray{T, N}) where {R, A, T <: Real, N}
    if !insupport(dist, x)
        return sum(xlogy.(one(eltype(dist.a)), zero(eltype(x))))
    end
    α = dist.a
    α0 = dist.α0
    s = sum(xlogy.(α .- 1, x); dims = 1)
    return sum(s .- dist.lmnB)
end

check_logpdf(::ExponentialFamilyDistribution{TensorDirichlet}, x::AbstractVector) = (MapBasedLogpdfCall(), x)
check_logpdf(::ExponentialFamilyDistribution{TensorDirichlet}, x) = (PointBasedLogpdfCall(), x)

BayesBase.default_prod_rule(::Type{<:TensorDirichlet}, ::Type{<:TensorDirichlet}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::TensorDirichlet, right::TensorDirichlet)
    return TensorDirichlet(left.a .+ right.a .- 1)
end

function BayesBase.insupport(dist::TensorDirichlet{T, N, A, Ts}, x::AbstractArray{T, N}) where {T, N, A, Ts}
    return size(dist) == size(x) && !any(x -> x < zero(x), x) && all(z -> z ≈ 1, sum(x; dims = 1))
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{TensorDirichlet}, x)
    l = getconditioner(ef)
    values = map(CartesianIndices(Base.tail(size(x)))) do i
        slice = @view x[:, i]
        sum(slice) ≈ 1 && all(y -> y > 0, slice)
    end
    return l == size(x) && all(values)
end

# Natural parametrization

function isproper(::NaturalParametersSpace, ::Type{TensorDirichlet}, η, conditioner)
    return length(η) > 1 && all(isless.(-1, η)) && all(!isinf, η) && all(!isnan, η)
end
function isproper(::MeanParametersSpace, ::Type{TensorDirichlet}, θ, conditioner)
    return length(θ) > 1 && all(>(0), θ) && all(!isinf, θ)
end

function (::MeanToNatural{TensorDirichlet})(tuple_of_θ::Tuple{Any}, _)
    (α,) = tuple_of_θ
    return (α - Ones{Float64}(size(α)),)
end

function (::NaturalToMean{TensorDirichlet})(tuple_of_η::Tuple{Any}, _)
    (η,) = tuple_of_η
    return (η + Ones{Float64}(size(η)),)
end

function getlogpartition(::NaturalParametersSpace, ::Type{TensorDirichlet}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_logpartition = getlogpartition(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector)
        result = zero(eltype(η))
        for i in 1:n_distributions
            idx_start = (i - 1) * k + 1
            idx_end = i * k
            @views params = η[idx_start:idx_end]
            result += dirichlet_logpartition(params)
        end

        return result
    end
end

function getgradlogpartition(
    ::NaturalParametersSpace,
    ::Type{TensorDirichlet},
    conditioner::NTuple{N, Int}
) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions

    # Get the "gradlogpartition" function for a standard Dirichlet
    dirichlet_gradlogpartition = getgradlogpartition(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector{T}) where {T}
        # Preallocate the output. We know we need `k * n_distributions` entries,
        # of the same element type as `η`.
        out = Vector{T}(undef, k * n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For the i-th distribution, grab the slice of η
                # and apply the Dirichlet gradlogpartition.
                out[(i-1)*k+1:i*k] = dirichlet_gradlogpartition(
                    @view η[(i-1)*k+1:i*k]
                )
            end
        end
        return out
    end
end

function getfisherinformation(::NaturalParametersSpace, ::Type{TensorDirichlet}, conditioner)
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_fisher = getfisherinformation(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector)
        blocks = Vector{Matrix{Float64}}(undef, n_distributions)

        for i in 1:n_distributions
            idx_start = (i - 1) * k + 1
            idx_end = i * k
            @views params = η[idx_start:idx_end]
            blocks[i] = dirichlet_fisher(params)
        end

        return BlockDiagonal(blocks)
    end
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}, conditioner) =
    (η) -> begin
        return mapreduce(x -> getlogpartition(MeanParametersSpace(), Dirichlet)(x), +, η)
    end

function getgradlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_gradlogpartition = getgradlogpartition(MeanParametersSpace(), Dirichlet)

    return function (θ::AbstractVector{T}) where {T}
        # Preallocate the output
        out = Vector{T}(undef, k * n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For each distribution, compute its gradient
                out[(i-1)*k+1:i*k] = dirichlet_gradlogpartition(
                    @view θ[(i-1)*k+1:i*k]
                )
            end
        end
        return out
    end
end

function getfisherinformation(::MeanParametersSpace, ::Type{TensorDirichlet}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_fisher = getfisherinformation(MeanParametersSpace(), Dirichlet)

    return function (θ::AbstractVector{T}) where {T}
        # Create blocks for block diagonal matrix
        blocks = Vector{Matrix{Float64}}(undef, n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For each distribution, compute its Fisher information
                blocks[i] = dirichlet_fisher(
                    @view θ[(i-1)*k+1:i*k]
                )
            end
        end

        return BlockDiagonal(blocks)
    end
end
