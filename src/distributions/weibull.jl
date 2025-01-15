export Weibull

import Distributions: Weibull, params
using DomainSets
using SpecialFunctions: digamma
using HCubature
using StaticArrays

# NOTE: The product of two Weibull distributions is NOT a Weibull distribution.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Weibull}
    conditioner_left = getconditioner(left)
    conditioner_right = getconditioner(right)
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)
    supp = DomainSets.HalfLine()
    if conditioner_left == conditioner_right
        basemeasure = (x) -> x^(2 * (conditioner_left - 1))
        sufficientstatistics = ((x) -> x^(conditioner_left),)
        logpartition =
            (η) ->
                log(abs(first(η))^(1 / conditioner_left)) + loggamma(2 - 1 / conditioner_left) -
                2 * log(abs(first(η))) - log(conditioner_left)
        naturalparameters = η_left + η_right
        attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, supp)
        return ExponentialFamilyDistribution(
            Univariate,
            Continuous,
            naturalparameters,
            nothing,
            attributes
        )
    else
        basemeasure = (x) -> x^(conditioner_left + conditioner_right - 2)
        sufficientstatistics = (x -> x^conditioner_left, x -> x^conditioner_right)
        naturalparameters = vcat(η_left, η_right)
        logpartition =
            (η) -> log(
                first(
                    hquadrature(
                        x ->
                            basemeasure(tan(x * pi / 2)) * exp(η' * [tan(x * pi / 2)^conditioner_left, tan(x * pi / 2)^conditioner_right]) *
                            (pi / 2) * (1 / cos(x * pi / 2)^2),
                        0,
                        1
                    )
                )
            )
        attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, supp)
        return ExponentialFamilyDistribution(
            Univariate,
            Continuous,
            naturalparameters,
            nothing,
            attributes
        )
    end
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Weibull}, η, conditioner) =
    !isnan(conditioner) && !isinf(conditioner) && conditioner > 0 && length(η) === 1 && all(<(0), η) && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{Weibull}, θ, conditioner) =
    !isnan(conditioner) && !isinf(conditioner) && conditioner > 0 && length(θ) === 1 && all(>(0), θ) && all(!isinf, θ) && all(!isnan, θ)

function separate_conditioner(::Type{Weibull}, params)
    shape, scale = params
    return ((scale,), shape)
end

function join_conditioner(::Type{Weibull}, cparams, conditioner)
    (scale,) = cparams
    return (conditioner, scale)
end

function (::MeanToNatural{Weibull})(tuple_of_θ::Tuple{Any}, conditioner)
    (λ,) = tuple_of_θ
    return (-(1 / λ)^(conditioner),)
end

function (::NaturalToMean{Weibull})(tuple_of_η::Tuple{Any}, conditioner)
    (η,) = tuple_of_η
    return ((-η)^inv(-conditioner),)
end

function unpack_parameters(::Type{Weibull}, packed, _)
    return (first(packed),)
end

function unpack_parameters(::Type{Weibull}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{Weibull}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Weibull}, conditioner) = x -> x^(conditioner - 1)
getlogbasemeasure(::Type{Weibull}, conditioner) = x -> (conditioner - 1) * log(x)
getsufficientstatistics(::Type{Weibull}, conditioner) = (x -> x^conditioner,)

getlogpartition(::NaturalParametersSpace, ::Type{Weibull}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(Weibull, η)
    return -log(-η1) - log(conditioner)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Weibull}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(Weibull, η)
    return SA[-inv(η1);]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Weibull}, _) = (η) -> begin
    (η1,) = unpack_parameters(Weibull, η)
    SA[inv(η1)^2;;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Weibull}, k) = (θ) -> begin
    (λ,) = unpack_parameters(Weibull, θ)
    return SA[k / λ;]
end

getgradlogpartition(::MeanParametersSpace, ::Type{Weibull}, conditioner) = (θ) -> begin
    (λ,) = unpack_parameters(Weibull, θ)
    return SA[-inv(η1);]
end

getfisherinformation(::MeanParametersSpace, ::Type{Weibull}, k) = (θ) -> begin
    (λ,) = unpack_parameters(MeanParametersSpace(), Weibull, θ)
    γ = -digamma(1) # Euler-Mascheroni constant (see https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant)
    a11 = (1 - 2γ + γ^2 + π^2 / 6) / (k^2)
    a12 = (γ - 1) / λ
    a21 = a12
    a22 = k^2 / (λ^2)

    return SA[a11 a12; a21 a22]
end
