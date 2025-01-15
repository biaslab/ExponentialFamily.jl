export VonMises
using Distributions
import Distributions: VonMises, params
import SpecialFunctions: besselj0

BayesBase.vague(::Type{<:VonMises}) = VonMises(0.0, tiny)
BayesBase.default_prod_rule(::Type{<:VonMises}, ::Type{<:VonMises}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::VonMises, right::VonMises)
    μleft, κleft = params(left)
    μright, κright = params(right)

    a = κleft * cos(μleft) + κright * cos(μright)
    b = κleft * sin(μleft) + κright * sin(μright)

    R = sqrt(a^2 + b^2)
    α = asin(b / R)

    phase = ((μleft - asin(sin(μleft))) + (μright - asin(sin(μright)))) / pi

    return VonMises(α + π * phase, R)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: VonMises}
    conditionerleft = getconditioner(left)
    conditionerright = getconditioner(right)
    ηleft = getnaturalparameters(left)
    ηright = getnaturalparameters(right)
   
    return ExponentialFamilyDistribution(VonMises, ηright + ηleft , conditionerleft + conditionerright)
     
end

BayesBase.insupport(ef::ExponentialFamilyDistribution{T}, value) where {T <: VonMises} = insupport(convert(Distribution, ef), value)

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{VonMises}, η, conditioner) = !isnothing(conditioner) && length(η) === 2 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{VonMises}, θ, conditioner) =
    !isnothing(conditioner) && length(θ) === 2 && getindex(θ, 2) > 0 && all(!isinf, θ) && all(!isnan, θ)

## We record the conditioner otherwise it is not possible to uniquely map back to mean paramaters space
function separate_conditioner(::Type{VonMises}, params)
    μ, κ = params
    return ((μ, κ), (μ - asin(sin(μ))) / pi)
end

join_conditioner(::Type{VonMises}, cparams, _) = cparams

function (::MeanToNatural{VonMises})(tuple_of_θ::Tuple{Any, Any}, _)
    (μ, κ) = tuple_of_θ
    return (κ * cos(μ), κ * sin(μ))
end

function (::NaturalToMean{VonMises})(tuple_of_η::Tuple{Any, Any}, conditioner)
    (η1, η2) = tuple_of_η
    κ = sqrt(η1^2 + η2^2)
    μ = asin(η2 / κ)
    return (conditioner * π + μ, κ)
end

function unpack_parameters(::Type{VonMises}, packed, _)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

function unpack_parameters(::Type{VonMises}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

isbasemeasureconstant(::Type{VonMises}) = ConstantBaseMeasure()

getbasemeasure(::Type{VonMises}, _) = (x) -> inv(twoπ)
getlogbasemeasure(::Type{VonMises}, _) = (x) -> -log(twoπ)
getsufficientstatistics(::Type{VonMises}, _) = (cos, sin)
getgradlogpartition(::NaturalParametersSpace, ::Type{VonMises}, _) = (η) -> begin
    u = sqrt(dot(η, η))
    same_part = besseli(1, u) / (u * besseli(0, u))
    return SA[η[1]*same_part, η[2]*same_part]
end
getlogpartition(::NaturalParametersSpace, ::Type{VonMises}, _) = (η) -> begin
    return log(besseli(0, sqrt(dot(η, η))))
end

getfisherinformation(::NaturalParametersSpace, ::Type{VonMises}, _) =
    (η) -> begin
        u = sqrt(dot(η, η))
        (η1, η2) = unpack_parameters(VonMises, η)
        bessel0 = besseli(0, u)
        bessel1 = besseli(1, u)
        bessel2 = (1 / 2) * (besseli(0, u) + besseli(2, u))

        h11 =
            (bessel2 / bessel0) * (η1^2 / u^2) - (bessel1 / bessel0)^2 * (η1^2 / u^2) +
            (bessel1 / bessel0) * (1 / u - (η1^2 / u^3))
        h22 =
            (bessel2 / bessel0) * (η2^2 / u^2) - (bessel1 / bessel0)^2 * (η2^2 / u^2) +
            (bessel1 / bessel0) * (1 / u - (η2^2 / u^3))
        h12 = (η1 * η2 / u^2) * (bessel2 / bessel0 - (bessel1 / bessel0)^2 - bessel1 / (u * bessel0))

        return SA[h11 h12; h12 h22]
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{VonMises}, _) = (θ) -> begin
    (_, κ) = unpack_parameters(VonMises, θ)
    return log(besseli(0, κ))
end

getfisherinformation(::MeanParametersSpace, ::Type{VonMises}, _) = (θ) -> begin
    (_, k) = unpack_parameters(VonMises, θ)
    bessel0 = besseli(0, k)
    bessel1 = besseli(1, k)
    bessel2 = (1 / 2) * (besseli(0, k) + besseli(2, k))
    return SA[(k)*bessel1/bessel0 0.0; 0.0 bessel2/bessel0-(bessel1/bessel0)^2]
end
