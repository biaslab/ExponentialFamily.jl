export Laplace
import Distributions: Laplace, params
using DomainSets

vague(::Type{<:Laplace}) = Laplace(0.0, huge)

prod_closed_rule(::Type{<:Laplace}, ::Type{<:Laplace}) = ClosedProd()

function Base.prod(::ClosedProd, left::Laplace, right::Laplace)
    location_left, scale_left = params(left)
    location_right, scale_right = params(right)

    if location_left == location_right
        return Laplace(location_left, scale_left * scale_right / (scale_left + scale_right))
    else
        ef_left = convert(KnownExponentialFamilyDistribution, left)
        ef_right = convert(KnownExponentialFamilyDistribution, right)

        (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
        (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
        basemeasure = (x) -> 1.0
        sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right)]
        sorted_conditioner = sort([conditioner_left, conditioner_right])
        function logpartition(η)
            A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right)
            A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right)
            A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right)
            B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
            B2 =
                (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
                (η[1] - η[2])
            B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

            return log(A1 * B1 + A2 * B2 + A3 * B3)
        end
        naturalparameters = [η_left, η_right]
        supp = support(left)

        return ExponentialFamilyDistribution(
            Float64,
            basemeasure,
            sufficientstatistics,
            naturalparameters,
            logpartition,
            supp
        )
    end
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Laplace)
    μ, θ = params(dist)
    return KnownExponentialFamilyDistribution(Laplace, [-inv(θ)], μ)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Laplace})
    return Laplace(getconditioner(exponentialfamily), -inv(first(getnaturalparameters(exponentialfamily))))
end

check_valid_natural(::Type{<:Laplace}, params) = length(params) == 1

check_valid_conditioner(::Type{<:Laplace}, conditioner) = true

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Laplace}) =
    first(getnaturalparameters(exponentialfamily)) < 0

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Laplace}) =
    log(-2 * first(getnaturalparameters(exponentialfamily)))
basemeasure(exponentialfamily::KnownExponentialFamilyDistribution{Laplace}, x) =
    1.0

basemeasure(d::Laplace, x) = 1.0