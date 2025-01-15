export Pareto

import Distributions: Pareto, shape, scale, params
using StaticArrays

BayesBase.vague(::Type{<:Pareto}) = Pareto(1e12)
BayesBase.cov(dist::Type{<:Pareto}) = var(dist)

# The default product between two `Pareto` objects is `PreserveTypeProd(Pareto)`,
# which is possible only if the location parameters match
BayesBase.default_prod_rule(::Type{<:Pareto}, ::Type{<:Pareto}) = PreserveTypeProd(Pareto)

function BayesBase.prod(::PreserveTypeProd{Pareto}, left::Pareto, right::Pareto)
    shape_left, scale_left = params(left)
    shape_right, scale_right = params(right)

    if isapprox(scale_left, scale_right)
        return Pareto(shape_left + shape_right + 1, scale_left)
    end

    error("""
        Cannot compute a closed product of two `Pareto` distribution with different location parameters.
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

# The default product between two `ExponentialFamilyDistribution{Pareto}` objects is 
# `ProdPreserveType(ExponentialFamilyDistribution{Pareto})`, which is possible only if the location parameters match
BayesBase.default_prod_rule(::Type{<:ExponentialFamilyDistribution{T}}, ::Type{<:ExponentialFamilyDistribution{T}}) where {T <: Pareto} =
    PreserveTypeProd(ExponentialFamilyDistribution{Pareto})

function BayesBase.prod!(
    container::ExponentialFamilyDistribution{Pareto},
    left::ExponentialFamilyDistribution{Pareto},
    right::ExponentialFamilyDistribution{Pareto}
)
    (η_container, conditioner_container) = (getnaturalparameters(container), getconditioner(container))
    (η_left, conditioner_left) = (getnaturalparameters(left), getconditioner(left))
    (η_right, conditioner_right) = (getnaturalparameters(right), getconditioner(right))

    if isapprox(conditioner_left, conditioner_right) && isapprox(conditioner_left, conditioner_container)
        LoopVectorization.vmap!(+, η_container, η_left, η_right)
        return container
    end

    error("""
        Cannot compute a closed product of two `Pareto` distribution in their natural parametrization with different conditioners (location parameter).
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution{Pareto}},
    left::ExponentialFamilyDistribution{Pareto},
    right::ExponentialFamilyDistribution{Pareto}
)
    return prod!(similar(left), left, right)
end
function BayesBase.insupport(ef::ExponentialFamilyDistribution{Pareto}, x)
    return x ∈ ClosedInterval(getconditioner(ef), Inf)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    ef_left::ExponentialFamilyDistribution{T},
    ef_right::ExponentialFamilyDistribution{T}
) where {T <: Pareto}
    (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
    (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
    if isapprox(conditioner_left, conditioner_right)
        return ExponentialFamilyDistribution(Pareto, η_left + η_right, conditioner_left)
    else
        basemeasure = (x) -> one(x)
        sufficientstatistics = (log,)
        naturalparameters = η_left + η_right
        support = RealInterval{Float64}(max(conditioner_left, conditioner_right), Inf)

        function logpartition(η)
            return dot(first(η) + 1, log(support.lb)) - log(-(first(η) + 1))
        end
        attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, support)
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

function isproper(::NaturalParametersSpace, ::Type{Pareto}, η, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(η) !== 1 || conditioner < 0
        return false
    end

    (η₁,) = unpack_parameters(Pareto, η)

    return !isnan(η₁) && !isinf(η₁) && η₁ < -1
end

function isproper(::MeanParametersSpace, ::Type{Pareto}, θ, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(θ) !== 1 || conditioner < 0
        return false
    end

    (shape,) = unpack_parameters(Pareto, θ)

    return !isnan(shape) && !isinf(shape) && shape > 0
end

function separate_conditioner(::Type{Pareto}, params)
    shape, scale = params
    return ((shape,), scale)
end

function join_conditioner(::Type{Pareto}, cparams, conditioner)
    (shape,) = cparams
    scale = conditioner
    return (shape, scale)
end

function (::MeanToNatural{Pareto})(tuple_of_θ::Tuple{Any}, _)
    (shape,) = tuple_of_θ
    return (-shape - one(shape),)
end

function (::NaturalToMean{Pareto})(tuple_of_η::Tuple{Any}, _)
    (η₁,) = tuple_of_η
    return (-η₁ - one(η₁),)
end

function unpack_parameters(::Type{Pareto}, packed, _)
    return (first(packed),)
end

function unpack_parameters(::Type{Pareto}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{Pareto}) = ConstantBaseMeasure()

getbasemeasure(::Type{Pareto}, _) = (x) -> oneunit(x)
getlogbasemeasure(::Type{Pareto}, _) = (x) -> zero(x)
getsufficientstatistics(::Type{Pareto}, conditioner) = (log,)

getlogpartition(::NaturalParametersSpace, ::Type{Pareto}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(Pareto, η)
    return log(conditioner^(one(η1) + η1) / (-one(η1) - η1))
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Pareto}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(Pareto, η)
    return SA[log(conditioner) - one(η1) / (one(η1) + η1);]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Pareto}, _) = (η) -> begin
    (η1,) = unpack_parameters(Pareto, η)
    return SA[1 / (-1 - η1)^2;;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Pareto}, conditioner) = (θ) -> begin
    (shape,) = unpack_parameters(Pareto, θ)
    return -log(shape) - shape * log(conditioner)
end

getgradlogpartition(::MeanParametersSpace, ::Type{Pareto}, conditioner) = (θ) -> begin
    (shape,) = unpack_parameters(Pareto, θ)
    return SA[-inv(shape) - log(conditioner);]
end

getfisherinformation(::MeanParametersSpace, ::Type{Pareto}, conditioner) = (θ) -> begin
    (α,) = unpack_parameters(Pareto, θ)
    ### Below fisher information is problematic if α is larger than conditioner as Pareto 
    ### does not satisfy regularity conditions
    # return SA[1/α^2 -1/conditioner; -1/conditioner α/conditioner^2]
    return SA[1 / α^2;;]
end
