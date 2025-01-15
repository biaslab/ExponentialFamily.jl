export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision, MvGaussianMeanScalePrecision
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily
export JointNormal, JointGaussian

const GaussianMeanVariance            = NormalMeanVariance
const GaussianMeanPrecision           = NormalMeanPrecision
const GaussianWeighteMeanPrecision    = NormalWeightedMeanPrecision
const MvGaussianMeanCovariance        = MvNormalMeanCovariance
const MvGaussianMeanPrecision         = MvNormalMeanPrecision
const MvGaussianWeightedMeanPrecision = MvNormalWeightedMeanPrecision
const MvGaussianMeanScalePrecision    = MvNormalMeanScalePrecision

const UnivariateNormalDistributionsFamily{T}   = Union{NormalMeanPrecision{T}, NormalMeanVariance{T}, NormalWeightedMeanPrecision{T}, Normal{T}}
const MultivariateNormalDistributionsFamily{T} = Union{MvNormalMeanPrecision{T}, MvNormalMeanCovariance{T}, MvNormalWeightedMeanPrecision{T}, MvNormalMeanScalePrecision{T}, MvNormal{T}}
const NormalDistributionsFamily{T}             = Union{UnivariateNormalDistributionsFamily{T}, MultivariateNormalDistributionsFamily{T}}

const UnivariateGaussianDistributionsFamily   = UnivariateNormalDistributionsFamily
const MultivariateGaussianDistributionsFamily = MultivariateNormalDistributionsFamily
const GaussianDistributionsFamily             = NormalDistributionsFamily

import Base: prod, convert, ndims
import Random: rand!
import Distributions: logpdf
import StatsFuns: invsqrt2π

using LoopVectorization
using StatsFuns: log2π
using LinearAlgebra
using SpecialFunctions

# special functions for `Normal` and `FullNormal`

function BayesBase.weightedmean_invcov(dist::Normal)
    mean, var = mean_var(dist)
    invcov = inv(var)
    return invcov * mean, invcov
end

function BayesBase.mean_invcov(dist::Normal)
    mean, var = mean_var(dist)
    return mean, inv(var)
end

function BayesBase.weightedmean_invcov(dist::Union{FullNormal, MvNormal})
    mean, var = mean_cov(dist)
    invcov = cholinv(var)
    return invcov * mean, invcov
end

function BayesBase.mean_invcov(dist::Union{FullNormal, MvNormal})
    mean, cov = mean_cov(dist)
    return mean, cholinv(cov)
end

function BayesBase.mean_std(dist::Union{FullNormal, MvNormal})
    mean, cov = mean_cov(dist)
    return mean, cholsqrt(cov)
end

# Joint over multiple Gaussians

"""
    JointNormal{D, S}

`JointNormal` is an auxilary structure used for the joint marginal over Normally distributed variables.
`JointNormal` stores a vector with the original dimensionalities (ds), so statistics can later be re-separated.

Use `ExponentialFamily.getcomponent(joint, index)` to get a specific component of the joint distribution.

# Fields
- `dist`: joint distribution (typically just a big `MvNormal` distribution, but maybe a tuple of individual means and covariance matrices)
- `ds`: a tuple with the original dimensionalities of individual `Normal` distributions
- `ds[k] = (n,)` where `n` is an integer indicates `Multivariate` normal of size `n`
- `ds[k] = ()` indicates `Univariate` normal
"""
struct JointNormal{D, S}
    dist :: D
    ds   :: S
end

dimensionalities(joint::JointNormal) = joint.ds

BayesBase.mean_cov(joint::JointNormal) = mean_cov(joint, joint.dist, joint.ds)

# In case if `JointNormal` internal representation stores the actual distribution we simply returns its statistics
BayesBase.mean_cov(::JointNormal, dist::NormalDistributionsFamily, ::Tuple) = mean_cov(dist)

# In case if `JointNormal` internal representation stores the actual distribution with a single univariate element we return its statistics as numbers
BayesBase.mean_cov(::JointNormal, dist::NormalDistributionsFamily, ::Tuple{Tuple{}}) = first.(mean_cov(dist))

# In case if `JointNormal` internal representation stores tuples of means and covariances we need to concatenate them
function BayesBase.mean_cov(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple)
    total = sum(prod.(ds); init = 0)
    @assert total !== 0 "Broken `JointNormal` state"

    T = promote_type(eltype.(first(dist))..., eltype.(last(dist))...)
    μ = zeros(T, total)
    Σ = zeros(T, total, total)

    sizes = prod.(ds)

    start = 1
    @inbounds for (index, size) in enumerate(sizes)
        dm, dc = first(dist)[index], last(dist)[index]
        μ[start:(start+size-1)] .= dm
        Σ[start:(start+size-1), start:(start+size-1)] .= dc
        start += size
    end

    return (μ, Σ)
end

# In case if `JointNormal` internal representation stores tuples of means and covariances with a single univariate element we return its statistics
function BayesBase.mean_cov(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple{Tuple})
    return (first(first(dist)), first(last(dist)))
end

BayesBase.entropy(joint::JointNormal) = entropy(joint, joint.dist)

BayesBase.entropy(joint::JointNormal, dist::NormalDistributionsFamily) = entropy(dist)
BayesBase.entropy(joint::JointNormal, dist::Tuple{Tuple, Tuple})       = entropy(convert(MvNormalMeanCovariance, mean_cov(joint)...))

Base.ndims(joint::JointNormal) = ndims(joint, joint.dist)

Base.ndims(joint::JointNormal, dist::NormalDistributionsFamily) = ndims(dist)
Base.ndims(joint::JointNormal, dist::Tuple{Tuple, Tuple})       = sum(length, first(dist))

BayesBase.paramfloattype(joint::JointNormal) = paramfloattype(joint, joint.dist)
BayesBase.convert_paramfloattype(::Type{T}, joint::JointNormal) where {T} = convert_paramfloattype(T, joint, joint.dist)

function BayesBase.paramfloattype(joint::JointNormal, dist::NormalDistributionsFamily)
    return paramfloattype(dist)
end

function BayesBase.convert_paramfloattype(::Type{T}, joint::JointNormal, dist::NormalDistributionsFamily) where {T}
    μ, Σ  = map(e -> convert_paramfloattype(T, e), mean_cov(dist))
    cdist = convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
    return JointNormal(cdist, joint.ds)
end

function Base.convert(::Type{JointNormal}, distribution::UnivariateNormalDistributionsFamily, sizes::Tuple{Tuple{}})
    return JointNormal(distribution, sizes)
end

function Base.convert(::Type{JointNormal}, distribution::MultivariateNormalDistributionsFamily, sizes::Tuple)
    return JointNormal(distribution, sizes)
end

function Base.convert(::Type{JointNormal}, means::Tuple, covs::Tuple)
    @assert length(means) === length(covs) "Cannot create the `JointNormal` with different number of statistics"
    return JointNormal((means, covs), size.(means))
end

# Return the marginalized statistics of the Gaussian corresponding to an index `index`
BayesBase.component(joint::JointNormal, index) = BayesBase.component(joint, joint.dist, joint.ds, joint.ds[index], index)

# `JointNormal` holds a single univariate gaussian and the dimensionalities indicate only a single Univariate element
function BayesBase.component(::JointNormal, dist::NormalMeanVariance, ds::Tuple{Tuple}, sz::Tuple{}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert size(dist) === sz "Broken `JointNormal` state"
    return dist
end

# `JointNormal` holds a single big gaussian and the dimensionalities indicate only a single Multivariate element
function BayesBase.component(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple{Tuple}, sz::Tuple{Int}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert size(dist) === sz "Broken `JointNormal` state"
    return dist
end

# `JointNormal` holds a single big gaussian and the dimensionalities indicate only a single Univariate element
function BayesBase.component(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple{Tuple}, sz::Tuple{}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert length(dist) === 1 "Broken `JointNormal` state"
    m, V = mean_cov(dist)
    return NormalMeanVariance(first(m), first(V))
end

# `JointNormal` holds a single big gaussian and the dimensionalities are generic, the element is Multivariate
function BayesBase.component(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple, sz::Tuple{Int}, index)
    @assert index <= length(ds) "Cannot marginalize `JointNormal` with single entry at index > number of elements"
    start = sum(prod.(ds[1:(index-1)]); init = 0) + 1
    len   = first(sz)
    stop  = start + len - 1
    μ, Σ  = mean_cov(dist)
    # Return the slice of the original `MvNormalMeanCovariance`
    return MvNormalMeanCovariance(view(μ, start:stop), view(Σ, start:stop, start:stop))
end

# `JointNormal` holds a single big gaussian and the dimensionalities are generic, the element is Univariate
function BayesBase.component(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple, sz::Tuple{}, index)
    @assert index <= length(ds) "Cannot marginalize `JointNormal` with single entry at index > number of elements"
    start = sum(prod.(ds[1:(index-1)]); init = 0) + 1
    μ, Σ = mean_cov(dist)
    # Return the slice of the original `MvNormalMeanCovariance`
    return NormalMeanVariance(μ[start], Σ[start, start])
end

# `JointNormal` holds gaussians individually, simply returns a Multivariate gaussian at index `index`
function BayesBase.component(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple, sz::Tuple{Int}, index)
    return MvNormalMeanCovariance(first(dist)[index], last(dist)[index])
end

# `JointNormal` holds gaussians individually, simply returns a Univariate gaussian at index `index`
function BayesBase.component(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple, sz::Tuple{}, index)
    return NormalMeanVariance(first(dist)[index], last(dist)[index])
end

# comparing JointNormals - similar to src/distributions/pointmass.jl
Base.isapprox(left::JointNormal, right::JointNormal; kwargs...) =
    isapprox(left.dist, right.dist; kwargs...) && left.ds == right.ds

"""An alias for the [`JointNormal`](@ref)."""
const JointGaussian = JointNormal

# Half-Normal related
function BayesBase.convert_paramfloattype(::Type{T}, distribution::Truncated{<:Normal}) where {T}
    return Truncated(convert_paramfloattype(T, distribution.untruncated), convert(T, distribution.lower), convert(T, distribution.upper))
end

# Variate forms promotion

BayesBase.promote_variate_type(::Type{Univariate}, ::Type{F}) where {F <: UnivariateNormalDistributionsFamily}     = F
BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{F}) where {F <: MultivariateNormalDistributionsFamily} = F

BayesBase.promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalMeanCovariance})        = NormalMeanVariance
BayesBase.promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalMeanPrecision})         = NormalMeanPrecision
BayesBase.promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalWeightedMeanPrecision}) = NormalWeightedMeanPrecision

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:NormalMeanVariance})          = MvNormalMeanCovariance
BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:NormalMeanPrecision})         = MvNormalMeanPrecision
BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:NormalWeightedMeanPrecision}) = MvNormalWeightedMeanPrecision

# Conversion to gaussian distributions from `Distributions.jl`

function Base.convert(::Type{Normal{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    mean, std = mean_std(dist)
    return Normal(convert(T, mean), convert(T, std))
end

function Base.convert(::Type{Normal}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(Normal{T}, dist)
end

function Base.convert(
    ::Type{MvNormal{T, C, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, C <: Distributions.PDMats.PDMat{T, Matrix{T}}, M <: AbstractVector{T}}
    mean, cov = mean_cov(dist)
    return MvNormal(convert(M, mean), Distributions.PDMats.PDMat(convert(AbstractMatrix{T}, cov)))
end

function Base.convert(::Type{MvNormal{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormal{T, Distributions.PDMats.PDMat{T, Matrix{T}}, Vector{T}}, dist)
end

function Base.convert(::Type{MvNormal}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormal{T}, dist)
end

function Base.convert(::Type{MvNormal{T}}, mean::AbstractVector, cov::AbstractMatrix) where {T <: Real}
    return MvNormal(convert(AbstractVector{T}, mean), Distributions.PDMats.PDMat(convert(AbstractMatrix{T}, cov)))
end

# Conversion to mean - variance parametrisation

function Base.convert(::Type{NormalMeanVariance{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    mean, var = mean_var(dist)
    return NormalMeanVariance(convert(T, mean), convert(T, var))
end

function Base.convert(::Type{MvNormalMeanCovariance{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormalMeanCovariance{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanCovariance{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanCovariance{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanCovariance{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    mean, cov = mean_cov(dist)
    return MvNormalMeanCovariance(convert(M, mean), convert(P, cov))
end

function Base.convert(::Type{NormalMeanVariance}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(NormalMeanVariance{T}, dist)
end

function Base.convert(::Type{MvNormalMeanCovariance}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormalMeanCovariance{T}, dist)
end

# Conversion to mean - precision parametrisation

function Base.convert(::Type{NormalMeanPrecision{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    mean, precision = mean_precision(dist)
    return NormalMeanPrecision(convert(T, mean), convert(T, precision))
end

function Base.convert(::Type{MvNormalMeanPrecision{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormalMeanPrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanPrecision{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanPrecision{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    mean, precision = mean_precision(dist)
    return MvNormalMeanPrecision(convert(M, mean), convert(P, precision))
end

function Base.convert(::Type{NormalMeanPrecision}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(NormalMeanPrecision{T}, dist)
end

function Base.convert(::Type{MvNormalMeanPrecision}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormalMeanPrecision{T}, dist)
end

# Conversion to weighted mean - precision parametrisation

function Base.convert(
    ::Type{NormalWeightedMeanPrecision{T}},
    dist::UnivariateNormalDistributionsFamily
) where {T <: Real}
    weightedmean, precision = weightedmean_precision(dist)
    return NormalWeightedMeanPrecision(convert(T, weightedmean), convert(T, precision))
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    weightedmean, precision = weightedmean_precision(dist)
    return MvNormalWeightedMeanPrecision(convert(M, weightedmean), convert(P, precision))
end

function Base.convert(
    ::Type{NormalWeightedMeanPrecision},
    dist::UnivariateNormalDistributionsFamily{T}
) where {T <: Real}
    return convert(NormalWeightedMeanPrecision{T}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision},
    dist::MultivariateNormalDistributionsFamily{T}
) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T}, dist)
end

# Special case for `Normal` to `NormalWeightedMeanPrecision` from `Distributions`

function Base.convert(::Type{NormalWeightedMeanPrecision}, dist::Normal)
    mean, var = mean_var(dist)
    precision = inv(var)
    return NormalWeightedMeanPrecision(precision * mean, precision)
end

# Special case for `FullNormal` to `NormalWeightedMeanPrecision` from `Distributions`

function Base.convert(::Type{MvNormalWeightedMeanPrecision}, dist::FullNormal)
    mean, cov = mean_cov(dist)
    precision = cholinv(cov)
    return MvNormalWeightedMeanPrecision(precision * mean, precision)
end

# isapprox

function Base.isapprox(left::UnivariateNormalDistributionsFamily, right::UnivariateNormalDistributionsFamily; kwargs...)
    return all(p -> isapprox(p[1], p[2]; kwargs...), zip(mean_var(left), mean_var(right)))
end

function Base.isapprox(left::D, right::D; kwargs...) where {D <: UnivariateNormalDistributionsFamily}
    return all(p -> isapprox(p[1], p[2]; kwargs...), zip(params(left), params(right)))
end

function Base.isapprox(left::MultivariateNormalDistributionsFamily, right::MultivariateNormalDistributionsFamily; kwargs...)
    return all(p -> isapprox(p[1], p[2]; kwargs...), zip(mean_cov(left), mean_cov(right)))
end

function Base.isapprox(left::D, right::D; kwargs...) where {D <: MultivariateNormalDistributionsFamily}
    return all(p -> isapprox(p[1], p[2]; kwargs...), zip(params(left), params(right)))
end

# Basic prod fallbacks to weighted mean precision and converts first argument back

BayesBase.default_prod_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:UnivariateNormalDistributionsFamily}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::L,
    right::R
) where {L <: UnivariateNormalDistributionsFamily, R <: UnivariateNormalDistributionsFamily}
    wleft  = convert(NormalWeightedMeanPrecision, left)
    wright = convert(NormalWeightedMeanPrecision, right)
    return prod(BayesBase.default_prod_rule(wleft, wright), wleft, wright)
end

function BayesBase.compute_logscale(
    ::N, left::L, right::R
) where {
    N <: UnivariateNormalDistributionsFamily,
    L <: UnivariateNormalDistributionsFamily,
    R <: UnivariateNormalDistributionsFamily
}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v                = v_left + v_right
    m                = m_left - m_right
    return -(logdet(v) + log2π) / 2 - m^2 / v / 2
end

BayesBase.default_prod_rule(::Type{<:MultivariateNormalDistributionsFamily}, ::Type{<:MultivariateNormalDistributionsFamily}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::L,
    right::R
) where {L <: MultivariateNormalDistributionsFamily, R <: MultivariateNormalDistributionsFamily}
    wleft  = convert(MvNormalWeightedMeanPrecision, left)
    wright = convert(MvNormalWeightedMeanPrecision, right)
    return prod(BayesBase.default_prod_rule(wleft, wright), wleft, wright)
end

function BayesBase.compute_logscale(
    ::N, left::L, right::R
) where {
    N <: MultivariateNormalDistributionsFamily,
    L <: MultivariateNormalDistributionsFamily,
    R <: MultivariateNormalDistributionsFamily
}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v                = v_left + v_right
    n                = length(left)
    v_inv, v_logdet  = cholinv_logdet(v)
    m                = m_left - m_right
    return -(v_logdet + n * log2π) / 2 - dot3arg(m, v_inv, m) / 2
end

BayesBase.logpdf_optimized(dist::UnivariateNormalDistributionsFamily) = convert(NormalMeanPrecision, dist)
BayesBase.logpdf_optimized(dist::MultivariateNormalDistributionsFamily) = convert(MvNormalMeanPrecision, dist)

BayesBase.sampling_optimized(dist::UnivariateNormalDistributionsFamily) = convert(Normal, dist)
BayesBase.sampling_optimized(dist::MultivariateNormalDistributionsFamily) = convert(MvNormal, dist)

# Sample related

## Univariate case

function BayesBase.rand(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily{T}) where {T}
    μ, σ = mean_std(dist)
    return μ + σ * randn(rng, T)
end

function BayesBase.rand(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily{T}, size::Int64) where {T}
    container = Vector{T}(undef, size)
    return rand!(rng, dist, container)
end

function BayesBase.rand!(
    rng::AbstractRNG,
    dist::UnivariateNormalDistributionsFamily,
    container::AbstractArray{T}
) where {T <: Real}
    randn!(rng, container)
    μ, σ = mean_std(dist)
    @turbo for i in eachindex(container)
        container[i] = μ + σ * container[i]
    end
    container
end

## Multivariate case

function BayesBase.rand(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily{T}) where {T}
    μ, L = mean_std(dist)
    return μ + L * randn(rng, T, length(μ))
end

function BayesBase.rand(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily{T}, size::Int64) where {T}
    container = Matrix{T}(undef, length(dist), size)
    return rand!(rng, dist, container)
end

function BayesBase.rand!(
    rng::AbstractRNG,
    dist::MultivariateNormalDistributionsFamily,
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

## Natural parameters for the Normal family distribution

### Univariate case

# Assume a single exponential family type tag both for all members of `UnivariateNormalDistributionsFamily`
# Thus all convert to `ExponentialFamilyDistribution{NormalMeanVariance}`
exponential_family_typetag(::UnivariateNormalDistributionsFamily) = NormalMeanVariance

BayesBase.params(::MeanParametersSpace, dist::UnivariateNormalDistributionsFamily) = mean_var(dist)

function isproper(::MeanParametersSpace, ::Type{NormalMeanVariance}, θ, conditioner)
    if length(θ) !== 2
        return false
    end
    (θ₁, θ₂) = unpack_parameters(NormalMeanVariance, θ)
    return isnothing(conditioner) && (!isinf(θ₁) && !isnan(θ₁)) && (θ₂ > 0)
end

function isproper(::NaturalParametersSpace, ::Type{NormalMeanVariance}, η, conditioner)
    if length(η) !== 2
        return false
    end
    (η₁, η₂) = unpack_parameters(NormalMeanVariance, η)
    return isnothing(conditioner) && (!isinf(η₁) && !isnan(η₁)) && (η₂ < 0)
end

function (::MeanToNatural{NormalMeanVariance})(tuple_of_θ::Tuple{Any, Any})
    (μ, σ²) = tuple_of_θ
    return (μ / σ², -inv(2σ²))
end

function (::NaturalToMean{NormalMeanVariance})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    return (-η₁ / 2η₂, -inv(2η₂))
end

function unpack_parameters(::Type{NormalMeanVariance}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

getsupport(::ExponentialFamilyDistribution{NormalMeanVariance}) = RealNumbers()

isbasemeasureconstant(::Type{NormalMeanVariance}) = ConstantBaseMeasure()

getbasemeasure(::Type{NormalMeanVariance}) = (x) -> convert(typeof(x), invsqrt2π)
getsufficientstatistics(::Type{NormalMeanVariance}) = (identity, abs2)

getlogpartition(::NaturalParametersSpace, ::Type{NormalMeanVariance}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(NormalMeanVariance, η)
    return -abs2(η₁) / 4η₂ - log(-2η₂) / 2
end

getgradlogpartition(::NaturalParametersSpace, ::Type{NormalMeanVariance}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(NormalMeanVariance, η)
        return SA[-η₁*inv(η₂ * 2), abs2(η₁)/(4*abs2(η₂))-1/(2*η₂)]
    end

getfisherinformation(::NaturalParametersSpace, ::Type{NormalMeanVariance}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(NormalMeanVariance, η)
        return SA[
            -inv(2η₂) η₁/(2abs2(η₂))
            η₁/(2abs2(η₂)) inv(2abs2(η₂))-abs2(η₁)/(2(η₂^3))
        ]
    end

### Univariate / mean parameters space

getlogpartition(::MeanParametersSpace, ::Type{NormalMeanVariance}) = (θ) -> begin
    (μ, σ²) = unpack_parameters(NormalMeanVariance, θ)
    return μ / 2σ² + log(sqrt(σ²))
end

getgradlogpartition(::MeanParametersSpace, ::Type{NormalMeanVariance}) =
    (θ) -> begin
        (μ, σ²) = unpack_parameters(NormalMeanVariance, θ)
        return SA[μ/σ², -abs2(μ)/(2σ²^2)+1/σ²]
    end

getfisherinformation(::MeanParametersSpace, ::Type{NormalMeanVariance}) = (θ) -> begin
    (_, σ²) = unpack_parameters(NormalMeanVariance, θ)
    return SA[inv(σ²) 0; 0 inv(2 * (σ²^2))]
end

### Fallback for the entire family 

getbasemeasure(::Type{<:UnivariateNormalDistributionsFamily}) = getbasemeasure(NormalMeanVariance)
getsufficientstatistics(::Type{<:UnivariateNormalDistributionsFamily}) = getsufficientstatistics(NormalMeanVariance)
getlogpartition(space::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{<:UnivariateNormalDistributionsFamily}) =
    getlogpartition(space, NormalMeanVariance)
getfisherinformation(space::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{<:UnivariateNormalDistributionsFamily}) =
    getfisherinformation(space, NormalMeanVariance)

### Multivariate case

# Assume a single exponential family type tag both for all members of `UnivariateNormalDistributionsFamily`
# Thus all convert to `ExponentialFamilyDistribution{NormalMeanVariance}`
exponential_family_typetag(::MultivariateGaussianDistributionsFamily) = MvNormalMeanCovariance

BayesBase.params(::MeanParametersSpace, dist::MultivariateGaussianDistributionsFamily) = mean_cov(dist)

function isproper(::MeanParametersSpace, ::Type{MvNormalMeanCovariance}, θ, conditioner)
    k = div(-1 + isqrt(1 + 4 * length(θ)), 2)
    if length(θ) < 2 || (length(θ) !== (k + k^2))
        return false
    end
    (μ, Σ) = unpack_parameters(MvNormalMeanCovariance, θ)
    return isnothing(conditioner) && length(μ) === size(Σ, 1) && (size(Σ, 1) === size(Σ, 2)) && isposdef(Σ)
end

function isproper(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance}, η, conditioner)
    k = div(-1 + isqrt(1 + 4 * length(η)), 2)
    if length(η) < 2 || (length(η) !== (k + k^2))
        return false
    end
    (η₁, η₂) = unpack_parameters(MvNormalMeanCovariance, η)
    return isnothing(conditioner) && length(η₁) === size(η₂, 1) && (size(η₂, 1) === size(η₂, 2)) && isposdef(-η₂)
end

function (::MeanToNatural{MvNormalMeanCovariance})(tuple_of_θ::Tuple{Any, Any})
    (μ, Σ) = tuple_of_θ
    Σ⁻¹ = cholinv(Σ)
    return (Σ⁻¹ * μ, Σ⁻¹ / -2)
end

function (::NaturalToMean{MvNormalMeanCovariance})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    Σ = cholinv(-2η₂)
    return (Σ * η₁, Σ)
end

function unpack_parameters(::Type{MvNormalMeanCovariance}, packed)
    len = length(packed)
    n = div(-1 + isqrt(1 + 4 * len), 2)

    p₁ = view(packed, 1:n)
    p₂ = reshape(view(packed, n+1:len), n, n)

    return (p₁, p₂)
end

# getsupport(ef::ExponentialFamilyDistribution{MvNormalMeanCovariance}) = RealNumbers()^div(-1 + isqrt(1 + 4 * length(getnaturalparameters(ef))), 2)
# The function above is not type-stable, the function below is type-stable, but does not uses an arbitrary `IndicatorFunction`
struct MvNormalDomainIndicator
    dims::Int
end

(indicator::MvNormalDomainIndicator)(v) = false
(indicator::MvNormalDomainIndicator)(v::AbstractVector) = length(v) === indicator.dims && isreal(v)

getsupport(ef::ExponentialFamilyDistribution{MvNormalMeanCovariance}) =
    Domain(IndicatorFunction{AbstractVector}(MvNormalDomainIndicator(div(-1 + isqrt(1 + 4 * length(getnaturalparameters(ef))), 2))))

isbasemeasureconstant(::Type{MvNormalMeanCovariance}) = ConstantBaseMeasure()

# It is a constant base measure with respect to `x`, only depends on its length, but we consider the length fixed
getbasemeasure(::Type{MvNormalMeanCovariance}) = (x) -> (2π)^(length(x) / -2)
getsufficientstatistics(::Type{MvNormalMeanCovariance}) = (identity, (x) -> x * x')

getlogpartition(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(MvNormalMeanCovariance, η)
    k = length(η₁)
    Cinv, l = cholinv_logdet(-η₂)
    return (dot(η₁, Cinv, η₁) / 2 - (k * log(2) + l)) / 2
end

getgradlogpartition(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(MvNormalMeanCovariance, η)
        Cinv, _ = cholinv_logdet(-η₂)
        return pack_parameters(MvNormalMeanCovariance, (0.5 * Cinv * η₁, 0.25 * Cinv * η₁ * η₁' * Cinv + 0.5 * Cinv))
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(MvNormalMeanCovariance, η)
        invη2 = -cholinv(-η₂)
        n = size(η₁, 1)
        ident = Eye(n)
        Iₙ = PermutationMatrix(1, 1)
        offdiag =
            1 / 4 * (invη2 * kron(ident, transpose(invη2 * η₁)) + invη2 * kron(η₁' * invη2, ident)) *
            kron(ident, kron(Iₙ, ident))
        G =
            -1 / 4 *
            (
                kron(invη2, invη2) * kron(ident, η₁) * kron(ident, transpose(invη2 * η₁)) +
                kron(invη2, invη2) * kron(η₁, ident) * kron(η₁' * invη2, ident)
            ) * kron(ident, kron(Iₙ, ident)) + 1 / 2 * kron(invη2, invη2)
        [-1/2*invη2 offdiag; offdiag' G]
    end

function PermutationMatrix(m, n)
    P = Matrix{Int}(undef, m * n, m * n)
    for i in 1:m*n
        for j in 1:m*n
            if j == 1 + m * (i - 1) - (m * n - 1) * floor((i - 1) / n)
                P[i, j] = 1
            else
                P[i, j] = 0
            end
        end
    end
    P
end

getfisherinformation(::MeanParametersSpace, ::Type{MvNormalMeanCovariance}) = (θ) -> begin
    μ, Σ = unpack_parameters(MvNormalMeanCovariance, θ)
    invΣ = cholinv(Σ)
    n = size(μ, 1)
    offdiag = zeros(n, n^2)
    G = (1 / 2) * kron(invΣ, invΣ)
    [invΣ offdiag; offdiag' G]
end
