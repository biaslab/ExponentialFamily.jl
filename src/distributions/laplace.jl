export Laplace
using Distributions
import Distributions: Laplace, params, logpdf
using DomainSets
using StaticArrays

BayesBase.vague(::Type{<:Laplace}) = Laplace(0.0, huge)

# The default product between two `Laplace` objects is `PreserveTypeProd(Laplace)`,
# which is possible only if the location parameters match
BayesBase.default_prod_rule(::Type{<:Laplace}, ::Type{<:Laplace}) = PreserveTypeProd(Laplace)

function BayesBase.prod(::PreserveTypeProd{Laplace}, left::Laplace, right::Laplace)
    location_left, scale_left = params(left)
    location_right, scale_right = params(right)

    if isapprox(location_left, location_right)
        return Laplace(location_left, scale_left * scale_right / (scale_left + scale_right))
    end

    error("""
        Cannot compute a closed product of two `Laplace` distribution with different location parameters.
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

# The default product between two `ExponentialFamilyDistribution{Laplace}` objects is 
# `ProdPreserveType(ExponentialFamilyDistribution{Laplace})`, which is possible only if the location parameters match
BayesBase.default_prod_rule(::Type{<:ExponentialFamilyDistribution{T}}, ::Type{<:ExponentialFamilyDistribution{T}}) where {T <: Laplace} =
    PreserveTypeProd(ExponentialFamilyDistribution{Laplace})

function BayesBase.prod!(
    container::ExponentialFamilyDistribution{Laplace},
    left::ExponentialFamilyDistribution{Laplace},
    right::ExponentialFamilyDistribution{Laplace}
)
    (η_container, conditioner_container) = (getnaturalparameters(container), getconditioner(container))
    (η_left, conditioner_left) = (getnaturalparameters(left), getconditioner(left))
    (η_right, conditioner_right) = (getnaturalparameters(right), getconditioner(right))

    if isapprox(conditioner_left, conditioner_right) && isapprox(conditioner_left, conditioner_container)
        LoopVectorization.vmap!(+, η_container, η_left, η_right)
        return container
    end

    error("""
        Cannot compute a closed product of two `Laplace` distribution in their natural parametrization with different conditioners (location parameter).
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution{Laplace}},
    left::ExponentialFamilyDistribution{Laplace},
    right::ExponentialFamilyDistribution{Laplace}
)
    return prod!(similar(left), left, right)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    ef_left::ExponentialFamilyDistribution{T},
    ef_right::ExponentialFamilyDistribution{T}
) where {T <: Laplace}
    (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
    (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
    if isapprox(conditioner_left, conditioner_right)
        return error(
            """
    To compute a generic product of two `ExponentialFamilyDistribution{Laplace}` distribution in their natural parametrization with same conditioners (location parameter).
    To compute a generic product in the natural parameters space, convert both distributions to the 
    `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution{Laplace})`
    prod strategy.
"""
        )
    else
        basemeasure = (x) -> one(x)
        vec_conditioner = [conditioner_left, conditioner_right]
        vec_params = vcat(η_left, η_right)
        sorted_conditioner = sort(vec_conditioner)
        naturalparameters = vec_params[indexin(sorted_conditioner, vec_conditioner)]
        μlarge = getindex(sorted_conditioner, 2)
        μsmall = first(sorted_conditioner)
        sufficientstatistics = (x -> abs(x - μsmall), x -> abs(x - μlarge))
        supp = RealInterval{Float64}(-Inf, Inf)
        ### η is in one-to-one relation with sorted conditioner
        function logpartition(η)
            A = exp(μsmall * η[1] + μlarge * η[2])
            B = exp(μlarge * η[2] - μsmall * η[1])
            C = exp(-μsmall * η[1] - μlarge * η[2])

            term1 = (A / (-η[1] - η[2])) * exp(μsmall * (-η[1] - η[2]))
            term2 = (B / (η[1] - η[2])) * (exp(μlarge * (η[1] - η[2])) - exp(μsmall * (η[1] - η[2])))
            term3 = (C / (-η[1] - η[2])) * exp(μlarge * (η[1] + η[2]))

            return log(term1 + term2 + term3)
        end
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

function isproper(::NaturalParametersSpace, ::Type{Laplace}, η, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(η) !== 1
        return false
    end

    (η₁,) = unpack_parameters(Laplace, η)

    return !isnan(η₁) && !isinf(η₁) && η₁ < 0
end

function isproper(::MeanParametersSpace, ::Type{Laplace}, θ, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(θ) !== 1
        return false
    end

    (scale,) = unpack_parameters(Laplace, θ)

    return !isnan(scale) && !isinf(scale) && scale > 0
end

function separate_conditioner(::Type{Laplace}, params)
    location, scale = params
    return ((scale,), location)
end

function join_conditioner(::Type{Laplace}, cparams, conditioner)
    (scale,) = cparams
    location = conditioner
    return (location, scale)
end

function (::MeanToNatural{Laplace})(tuple_of_θ::Tuple{Any}, _)
    (scale,) = tuple_of_θ
    return (-inv(scale),)
end

function (::NaturalToMean{Laplace})(tuple_of_η::Tuple{Any}, _)
    (η₁,) = tuple_of_η
    return (-inv(η₁),)
end

function unpack_parameters(::Type{Laplace}, packed, _)
    return (first(packed),)
end

function unpack_parameters(::Type{Laplace}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{Laplace}) = ConstantBaseMeasure()

getbasemeasure(::Type{Laplace}, _) = (x) -> oneunit(x)
getlogbasemeasure(::Type{Laplace}, _) = (x) -> zero(x)

getsufficientstatistics(::Type{Laplace}, conditioner) = (
    (x) -> abs(x - conditioner),
)

getlogpartition(::NaturalParametersSpace, ::Type{Laplace}, _) = (η) -> begin
    (η₁,) = unpack_parameters(Laplace, η)
    return log(-2 / η₁)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Laplace}, _) = (η) -> begin
    (η₁,) = unpack_parameters(Laplace, η)
    return SA[-inv(η₁);]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Laplace}, _) = (η) -> begin
    (η₁,) = unpack_parameters(Laplace, η)
    return SA[inv(η₁^2);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Laplace}, _) = (θ) -> begin
    (scale,) = unpack_parameters(Laplace, θ)
    return log(2scale)
end

getfisherinformation(::MeanParametersSpace, ::Type{Laplace}, _) = (θ) -> begin
    (scale,) = unpack_parameters(Laplace, θ)
    return SA[inv(abs2(scale));;] # 1 / scale^2
end
