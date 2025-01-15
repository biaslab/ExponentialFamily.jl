export MvNormalMeanScalePrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

"""
    MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}} <: AbstractMvNormal

A multivariate normal distribution with mean `μ` and scale parameter `γ` that scales the identity precision matrix.

# Type Parameters
- `T`: The element type of the mean vector and scale parameter
- `M`: The type of the mean vector, which must be a subtype of `AbstractVector{T}`

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution
- `γ::T`: The scale parameter that scales the identity precision matrix

# Notes
The precision matrix of this distribution is `γ * I`, where `I` is the identity matrix.
The covariance matrix is the inverse of the precision matrix, i.e., `(1/γ) * I`.
"""
struct MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}} <: AbstractMvNormal
    μ::M
    γ::T
end

function MvNormalMeanScalePrecision(μ::AbstractVector{<:Real}, γ::Real)
    T = promote_type(eltype(μ), eltype(γ))
    return MvNormalMeanScalePrecision(convert(AbstractArray{T}, μ), convert(T, γ))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{<:Integer}, γ::Real)
    return MvNormalMeanScalePrecision(float.(μ), float(γ))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{T}) where {T}
    return MvNormalMeanScalePrecision(μ, convert(T, 1))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{T1}, γ::T2) where {T1, T2}
    T = promote_type(T1, T2)
    μ_new = convert(AbstractArray{T}, μ)
    γ_new = convert(T, γ)(length(μ))
    return MvNormalMeanScalePrecision(μ_new, γ_new)
end

function unpack_parameters(::Type{MvNormalMeanScalePrecision}, packed)
    p₁ = view(packed, 1:length(packed)-1)
    p₂ = packed[end]

    return (p₁, p₂)
end

function isproper(::NaturalParametersSpace, ::Type{MvNormalMeanScalePrecision}, η, conditioner)
    k = length(η) - 1
    if length(η) < 2 || (length(η) !== k + 1)
        return false
    end
    (η₁, η₂) = unpack_parameters(MvNormalMeanScalePrecision, η)
    return isnothing(conditioner) && isone(size(η₂, 1)) && isposdef(-η₂)
end

function (::MeanToNatural{MvNormalMeanScalePrecision})(tuple_of_θ::Tuple{Any, Any})
    (μ, γ) = tuple_of_θ
    return (γ * μ, -γ / 2)
end

function (::NaturalToMean{MvNormalMeanScalePrecision})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    γ = -2 * η₂
    return (η₁ / γ, γ)
end

function nabs2(x)
    return sum(map(abs2, x))
end

getsufficientstatistics(::Type{MvNormalMeanScalePrecision}) = (identity, nabs2)

# Conversions
function Base.convert(
    ::Type{MvNormal{T, C, M}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real, C <: Distributions.PDMats.PDMat{T, Matrix{T}}, M <: AbstractVector{T}}
    m, σ = mean(dist), std(dist)
    return MvNormal(convert(M, m), convert(T, σ))
end

function Base.convert(
    ::Type{MvNormalMeanScalePrecision{T, M}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real, M <: AbstractArray{T}}
    m, γ = mean(dist), dist.γ
    return MvNormalMeanScalePrecision{T, M}(convert(M, m), convert(T, γ))
end

function Base.convert(
    ::Type{MvNormalMeanScalePrecision{T}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real}
    return convert(MvNormalMeanScalePrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(::Type{MvNormalMeanCovariance}, dist::MvNormalMeanScalePrecision)
    m, σ = mean(dist), cov(dist)
    return MvNormalMeanCovariance(m, σ * diagm(ones(length(m))))
end

function Base.convert(::Type{MvNormalMeanPrecision}, dist::MvNormalMeanScalePrecision)
    m, γ = mean(dist), precision(dist)
    return MvNormalMeanPrecision(m, γ * diagm(ones(length(m))))
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision}, dist::MvNormalMeanScalePrecision)
    m, γ = mean(dist), precision(dist)
    return MvNormalWeightedMeanPrecision(γ * m, γ * diagm(ones(length(m))))
end

Distributions.distrname(::MvNormalMeanScalePrecision) = "MvNormalMeanScalePrecision"

BayesBase.weightedmean(dist::MvNormalMeanScalePrecision) = precision(dist) * mean(dist)

BayesBase.mean(dist::MvNormalMeanScalePrecision)      = dist.μ
BayesBase.mode(dist::MvNormalMeanScalePrecision)      = mean(dist)
BayesBase.var(dist::MvNormalMeanScalePrecision)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalMeanScalePrecision)       = cholinv(invcov(dist))
BayesBase.invcov(dist::MvNormalMeanScalePrecision)    = scale(dist) * I(length(mean(dist)))
BayesBase.std(dist::MvNormalMeanScalePrecision)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalMeanScalePrecision) = -chollogdet(invcov(dist))
BayesBase.scale(dist::MvNormalMeanScalePrecision)     = dist.γ
BayesBase.params(dist::MvNormalMeanScalePrecision)    = (mean(dist), scale(dist))

function Distributions.sqmahal(dist::MvNormalMeanScalePrecision, x::AbstractVector)
    T = promote_type(eltype(x), paramfloattype(dist))
    return sqmahal!(similar(x, T), dist, x)
end

function Distributions.sqmahal!(r, dist::MvNormalMeanScalePrecision, x::AbstractVector)
    μ, γ = params(dist)
    @inbounds @simd for i in 1:length(r)
        r[i] = μ[i] - x[i]
    end
    return dot3arg(r, γ, r) # x' * A * x
end

Base.eltype(::MvNormalMeanScalePrecision{T}) where {T} = T
Base.precision(dist::MvNormalMeanScalePrecision) = invcov(dist)
Base.length(dist::MvNormalMeanScalePrecision) = length(mean(dist))
Base.ndims(dist::MvNormalMeanScalePrecision) = length(dist)
Base.size(dist::MvNormalMeanScalePrecision) = (length(dist),)

Base.convert(::Type{<:MvNormalMeanScalePrecision}, μ::AbstractVector, γ::Real) = MvNormalMeanScalePrecision(μ, γ)

function Base.convert(::Type{<:MvNormalMeanScalePrecision{T}}, μ::AbstractVector, γ::Real) where {T <: Real}
    MvNormalMeanScalePrecision(convert(AbstractArray{T}, μ), convert(T, γ))
end

BayesBase.vague(::Type{<:MvNormalMeanScalePrecision}, dims::Int) =
    MvNormalMeanScalePrecision(zeros(Float64, dims), convert(Float64, tiny))

BayesBase.default_prod_rule(::Type{<:MvNormalMeanScalePrecision}, ::Type{<:MvNormalMeanScalePrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanScalePrecision, right::MvNormalMeanScalePrecision)
    w = scale(left) + scale(right)
    m = (scale(left) * mean(left) + scale(right) * mean(right)) / w
    return MvNormalMeanScalePrecision(m, w)
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalMeanScalePrecision{T}) where {T}
    μ, γ = params(dist)
    d = length(μ)
    return rand!(rng, dist, Vector{T}(undef, d))
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalMeanScalePrecision{T}, size::Int64) where {T}
    container = Matrix{T}(undef, length(dist), size)
    return rand!(rng, dist, container)
end

# FIXME: This is not the most efficient way to generate random samples within container
#        it needs to work with scale method, not with std
function BayesBase.rand!(
    rng::AbstractRNG,
    dist::MvNormalMeanScalePrecision,
    container::AbstractArray{T}
) where {T <: Real}
    preallocated = similar(container)
    randn!(rng, reshape(preallocated, length(preallocated)))
    μ, L = mean_std(dist)
    @views for i in axes(preallocated, 2)
        copyto!(container[:, i], μ)
        mul!(container[:, i], L, preallocated[:, i], 1, 1)
    end
    container
end

function getsupport(ef::ExponentialFamilyDistribution{MvNormalMeanScalePrecision})
    dim = length(getnaturalparameters(ef)) - 1
    return Domain(IndicatorFunction{AbstractVector}(MvNormalDomainIndicator(dim)))
end

isbasemeasureconstant(::Type{MvNormalMeanScalePrecision}) = ConstantBaseMeasure()

getbasemeasure(::Type{MvNormalMeanScalePrecision}) = (x) -> (2π)^(-length(x) / 2)

getlogbasemeasure(::Type{MvNormalMeanScalePrecision}) = (x) -> -length(x) / 2 * log2π

getlogpartition(::NaturalParametersSpace, ::Type{MvNormalMeanScalePrecision}) =
    (η) -> begin
        η1 = @view η[1:end-1]
        η2 = η[end]
        k = length(η1)
        Cinv = inv(η2)
        return -dot(η1, 1 / 4 * Cinv, η1) - (k / 2) * log(-2 * η2)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{MvNormalMeanScalePrecision}) =
    (η) -> begin
        η1 = @view η[1:end-1]
        η2 = η[end]
        inv2 = inv(η2)
        k = length(η1)
        return pack_parameters(MvNormalMeanCovariance, (-1 / (2 * η2) * η1, dot(η1, η1) / 4 * inv2^2 - k / 2 * inv2))
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MvNormalMeanScalePrecision}) =
    (η) -> begin
        η1 = @view η[1:end-1]
        η2 = η[end]
        k = length(η1)

        η1_part = -inv(2 * η2) * I(length(η1))
        η1η2 = zeros(k, 1)
        η1η2 .= η1 * inv(2 * η2^2)

        η2_part = k * inv(2abs2(η2)) - dot(η1, η1) / (2 * η2^3)

        return ArrowheadMatrix(η2_part, η1η2, diag(η1_part))
    end

getfisherinformation(::MeanParametersSpace, ::Type{MvNormalMeanScalePrecision}) =
    (θ) -> begin
        μ = @view θ[1:end-1]
        γ = θ[end]
        k = length(μ)

        matrix = zeros(eltype(μ), (k + 1))
        matrix[1:k] .= γ
        matrix[k+1] = k * inv(2abs2(γ))
        return Diagonal(matrix)
    end
