export Rayleigh

import Distributions: Rayleigh, params
using DomainSets
using StaticArrays

BayesBase.vague(::Type{<:Rayleigh}) = Rayleigh(Float64(huge))

# NOTE: The product of two Rayleigh distributions is NOT a Rayleigh distribution.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Rayleigh}
    η1 = getnaturalparameters(left)
    η2 = getnaturalparameters(right)
    naturalparameters = η1 + η2
    basemeasure = (x) -> 4 * x^2 / sqrt(pi)
    sufficientstatistics = (x -> x^2,)
    logpartition = (η) -> (-3 / 2)log(-first(η))
    support = DomainSets.HalfLine()
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, support)

    return ExponentialFamilyDistribution(
        Univariate,
        Continuous,
        naturalparameters,
        nothing,
        attributes
    )
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Rayleigh}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && all(<(0), η) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Rayleigh}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && all(>(0), θ) && all(!isinf, θ)

function (::MeanToNatural{Rayleigh})(tuple_of_θ::Tuple{Any})
    (σ,) = tuple_of_θ
    return (-1 / (2 * σ^2),)
end

function (::NaturalToMean{Rayleigh})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (sqrt(-1 / (2η)),)
end

unpack_parameters(::Type{Rayleigh}, packed) = (first(packed),)

isbasemeasureconstant(::Type{Rayleigh}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Rayleigh}) = identity
getsufficientstatistics(::Type{Rayleigh}) = (x -> x^2,)

getlogpartition(::NaturalParametersSpace, ::Type{Rayleigh}) = (η) -> begin
    (η1,) = unpack_parameters(Rayleigh, η)
    return -log(-2 * η1)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Rayleigh}) = (η) -> begin
    (η1,) = unpack_parameters(Rayleigh, η)
    return SA[-inv(η1);]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Rayleigh}) = (η) -> begin
    (η1,) = unpack_parameters(Rayleigh, η)
    SA[inv(η1^2);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Rayleigh}) = (θ) -> begin
    (σ,) = unpack_parameters(Rayleigh, θ)
    return 2 * log(σ)
end

getgradlogpartition(::MeanParametersSpace, ::Type{Rayleigh}) = (θ) -> begin
    (σ,) = unpack_parameters(Rayleigh, θ)
    return SA[2 / σ;]
end

getfisherinformation(::MeanParametersSpace, ::Type{Rayleigh}) = (θ) -> begin
    (σ,) = unpack_parameters(Rayleigh, θ)
    return SA[4 / σ^2;;]
end
