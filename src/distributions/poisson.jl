export Poisson

import SpecialFunctions: besseli
import Distributions: Poisson, shape, scale, cov
using DomainSets
using StaticArrays

BayesBase.cov(dist::Poisson) = var(dist)

# NOTE: The product of two Poisson distributions is NOT a Poisson distribution.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Poisson}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> one(x) / gamma(x + one(x))^2
    sufficientstatistics = (identity,)
    logpartition = (η) -> log(abs(besseli(0, 2 * exp(first(η) / 2))))
    supp = DomainSets.NaturalNumbers()
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, supp)

    return ExponentialFamilyDistribution(
        Univariate,
        Discrete,
        naturalparameters,
        nothing,
        attributes
    )
end

# Natural parametrization

getsupport(::Type{Poisson}) = DomainSets.NaturalNumbers()

isproper(::NaturalParametersSpace, ::Type{Poisson}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{Poisson}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && all(>(0), θ) && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{Poisson})(tuple_of_θ::Tuple{Any})
    (λ,) = tuple_of_θ
    return (log(λ),)
end

function (::NaturalToMean{Poisson})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (exp(η),)
end

unpack_parameters(::Type{Poisson}, packed) = (first(packed),)

isbasemeasureconstant(::Type{Poisson}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Poisson}) = (x) -> one(x) / gamma(x + one(x))
getlogbasemeasure(::Type{Poisson}) = (x) -> -loggamma(x + one(x))
getsufficientstatistics(::Type{Poisson}) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{Poisson}) = (η) -> begin
    (η1,) = unpack_parameters(Poisson, η)
    return exp(η1)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Poisson}) = (η) -> begin
    (η1,) = unpack_parameters(Poisson, η)
    return SA[exp(η1)]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Poisson}) = (η) -> begin
    (η1,) = unpack_parameters(Poisson, η)
    SA[exp(η1);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Poisson}) = (θ) -> begin
    (λ,) = unpack_parameters(Poisson, θ)
    return λ
end

getfisherinformation(::MeanParametersSpace, ::Type{Poisson}) = (θ) -> begin
    (λ,) = unpack_parameters(Poisson, θ)
    return SA[1 / λ;;]
end
