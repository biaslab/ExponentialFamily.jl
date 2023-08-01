export Binomial
using DomainSets
import Distributions: Binomial, probs, Univariate
import StatsFuns: logit, logistic, log1pexp

import HypergeometricFunctions: _₂F₁
import StaticArrays: SA

vague(::Type{<:Binomial}, trials::Int) = Binomial(trials)

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

function convert_eltype(::Type{Binomial}, ::Type{T}, distribution::Binomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Binomial(n, convert(AbstractVector{T}, p))
end

function insupport(ef::ExponentialFamilyDistribution{Binomial, P, C, Safe}, x) where {P, C}
    return x ∈ ClosedInterval{Int}(0, getconditioner(ef)) && typeof(x) <: Int
end

closed_prod_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::Binomial, right::Binomial)
    efleft = convert(ExponentialFamilyDistribution, left)
    efright = convert(ExponentialFamilyDistribution, right)

    return prod(efleft, efright)
end

# NOTE: The product of two Binomial distributions is NOT a Binomial distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Binomial}
    left_trials, right_trials = getconditioner(left), getconditioner(right)

    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right

    function basemeasure(x)
        p_left, p_right, p_x = promote(left_trials, right_trials, x)
        binomial_prod(p_left, p_right, p_x)
    end

    sufficientstatistics = (x) -> SA[x]
    logpartition = (η) -> log(_₂F₁(-left_trials, -right_trials, 1, exp(first(η))))
    supp = 0:max(left_trials, right_trials)

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        basemeasure,
        sufficientstatistics,
        logpartition,
        supp
    )
end

function pack_naturalparameters(dist::Binomial)
    return [logit(dist.p)]
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Binomial})
    vectorized = getnaturalparameters(ef)
    @inbounds η1 = vectorized[1]
    return (η1,)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Binomial)
    return ExponentialFamilyDistribution(Binomial, pack_naturalparameters(dist), dist.n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Binomial})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return Binomial(getconditioner(exponentialfamily), logistic(η))
end

check_valid_natural(::Type{<:Binomial}, params) = length(params) == 1

function check_valid_conditioner(::Type{<:Binomial}, conditioner)
    isinteger(conditioner) && conditioner > zero(conditioner)
end

isproper(exponentialfamily::ExponentialFamilyDistribution{Binomial}) =
    getconditioner(exponentialfamily) > zero(Int64) ? true : false

logpartition(exponentialfamily::ExponentialFamilyDistribution{Binomial}) =
    getconditioner(exponentialfamily)log1pexp(first(unpack_naturalparameters(exponentialfamily)))

function fisherinformation(dist::Binomial)
    n, p = params(dist)
    return SA[n / (p * (1 - p));;]
end

function fisherinformation(ef::ExponentialFamilyDistribution{Binomial})
    (η,) = unpack_naturalparameters(ef)
    aux = logistic(η)
    n = getconditioner(ef)

    return SA[n * aux * (1 - aux);;]
end

basemeasureconstant(::ExponentialFamilyDistribution{Binomial}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Binomial}) = NonConstantBaseMeasure()

basemeasure(ef::ExponentialFamilyDistribution{Binomial}) = x -> basemeasure(ef, x)
basemeasure(ef::ExponentialFamilyDistribution{Binomial}, x) = binomial(getconditioner(ef), x)

sufficientstatistics(type::Type{<:Binomial}) = x -> sufficientstatistics(type, x)
sufficientstatistics(::Type{<:Binomial}, x) = SA[x]
sufficientstatistics(ef::ExponentialFamilyDistribution{<:Binomial}) = x -> sufficientstatistics(ef, x)
sufficientstatistics(::ExponentialFamilyDistribution{<:Binomial}, x) = SA[x]
