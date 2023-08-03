export Rayleigh

import Distributions: Rayleigh, params
using DomainSets
using StaticArrays

vague(::Type{<:Rayleigh}) = Rayleigh(Float64(huge))

prod_analytical_rule(::Type{<:Rayleigh}, ::Type{<:Rayleigh}) = ClosedProd()

# NOTE: The product of two Rayleigh distributions is NOT a Rayleigh distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Rayleigh}
    η1 = getnaturalparameters(left)
    η2 = getnaturalparameters(right)
    naturalparameters = η1 + η2
    basemeasure = (x) -> 4 * x^2 / sqrt(pi)
    sufficientstatistics = (x) -> SA[x^2]
    logpartition = (η) -> (-3 / 2)log(η)
    support = DomainSets.HalfLine()

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        basemeasure,
        sufficientstatistics,
        logpartition,
        support
    )
end

function Base.prod(::ClosedProd, left::T, right::T) where {T <: Rayleigh}
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)
    return prod(ClosedProd(), ef_left, ef_right)
end

pack_naturalparameters(dist::Rayleigh) = [MINUSHALF / first(params(dist))^2]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Rayleigh})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return (η1,)
end

isproper(ef::ExponentialFamilyDistribution{Rayleigh}) = first(unpack_naturalparameters(ef)) < 0

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Rayleigh) =
    ExponentialFamilyDistribution(Rayleigh, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{Rayleigh})
    (η,) = unpack_naturalparameters(ef)
    return Rayleigh(sqrt(-1 / (2η)))
end

check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

logpartition(ef::ExponentialFamilyDistribution{Rayleigh}) = -log(-2 * first(unpack_naturalparameters(ef)))

fisherinformation(dist::Rayleigh) = SA[4 / scale(dist)^2;;]

fisherinformation(ef::ExponentialFamilyDistribution{Rayleigh}) = SA[inv(first(unpack_naturalparameters(ef))^2);;]

support(::ExponentialFamilyDistribution{Rayleigh}) = ClosedInterval{Real}(0, Inf)

basemeasureconstant(::ExponentialFamilyDistribution{Rayleigh}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Rayleigh}) = NonConstantBaseMeasure()

sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}) =
    (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    return SA[x^2]
end
basemeasure(ef::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}) = (x) -> basemeasure(ef, x)
function basemeasure(::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    return x
end