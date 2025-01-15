using ExponentialFamily, BayesBase, Distributions, Test, StatsFuns, BenchmarkTools, Random, FillArrays

import Distributions: RealInterval, ContinuousUnivariateDistribution, Univariate
import ExponentialFamily: basemeasure, logbasemeasure, sufficientstatistics, logpartition, insupport, ConstantBaseMeasure
import ExponentialFamily: getnaturalparameters, getbasemeasure, getlogbasemeasure, getsufficientstatistics, getlogpartition, getsupport
import ExponentialFamily: ExponentialFamilyDistributionAttributes, NaturalParametersSpace

# import ExponentialFamily:
#     ExponentialFamilyDistribution, getnaturalparameters, getconditioner, reconstructargument!, as_vec,
#     pack_naturalparameters, unpack_naturalparameters, insupport
# import Distributions: pdf, logpdf, cdf

## ===========================================================================
## Tests fixtures
const ArbitraryExponentialFamilyAttributes = ExponentialFamilyDistributionAttributes(
    (x) -> 1 / x,
    ((x) -> x, (x) -> log(x)),
    (η) -> 1 / sum(η),
    RealInterval(0, Inf)
)

# Arbitrary distribution (un-conditioned)
struct ArbitraryDistributionFromExponentialFamily <: ContinuousUnivariateDistribution
    p1::Float64
    p2::Float64
end

ExponentialFamily.isproper(::NaturalParametersSpace, ::Type{ArbitraryDistributionFromExponentialFamily}, η, conditioner) = isnothing(conditioner)
ExponentialFamily.isbasemeasureconstant(::Type{ArbitraryDistributionFromExponentialFamily}) = ConstantBaseMeasure()
ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (x) -> oneunit(x)
ExponentialFamily.getlogbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (x) -> zero(x)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ((x) -> x, (x) -> log(x))
ExponentialFamily.getlogpartition(::NaturalParametersSpace, ::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

BayesBase.vague(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ArbitraryDistributionFromExponentialFamily(1.0, 1.0)

BayesBase.params(dist::ArbitraryDistributionFromExponentialFamily) = (dist.p1, dist.p2)

(::MeanToNatural{ArbitraryDistributionFromExponentialFamily})(params::Tuple) = (params[1] + 1, params[2] + 1)
(::NaturalToMean{ArbitraryDistributionFromExponentialFamily})(params::Tuple) = (params[1] - 1, params[2] - 1)

ExponentialFamily.unpack_parameters(::Type{ArbitraryDistributionFromExponentialFamily}, η) = (η[1], η[2])
ExponentialFamily.unpack_parameters(::Type{ArbitraryDistributionFromExponentialFamily}, η, _) = (η[1], η[2])

# Arbitrary distribution (conditioned)
struct ArbitraryConditionedDistributionFromExponentialFamily <: ContinuousUnivariateDistribution
    con::Int
    p1::Float64
end

ExponentialFamily.isproper(::NaturalParametersSpace, ::Type{ArbitraryConditionedDistributionFromExponentialFamily}, η, conditioner) = isinteger(conditioner)
ExponentialFamily.isbasemeasureconstant(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) = NonConstantBaseMeasure()
ExponentialFamily.getbasemeasure(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) = (x) -> x^conditioner
ExponentialFamily.getlogbasemeasure(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) = (x) -> conditioner * log(x)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) =
    ((x) -> log(x - conditioner),)
ExponentialFamily.getlogpartition(::NaturalParametersSpace, ::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) =
    (η) -> conditioner / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) = RealInterval(0, Inf)

BayesBase.vague(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) =
    ArbitraryConditionedDistributionFromExponentialFamily(1.0, -2)

BayesBase.params(dist::ArbitraryConditionedDistributionFromExponentialFamily) = (dist.con, dist.p1)
BayesBase.value_support(::ArbitraryConditionedDistributionFromExponentialFamily) = Continuous

ExponentialFamily.separate_conditioner(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, params) = ((params[2],), params[1])
ExponentialFamily.join_conditioner(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, cparams, conditioner) = (conditioner, cparams...)

(::MeanToNatural{ArbitraryConditionedDistributionFromExponentialFamily})(params::Tuple, conditioner::Number) = (params[1] + conditioner,)
(::NaturalToMean{ArbitraryConditionedDistributionFromExponentialFamily})(params::Tuple, conditioner::Number) = (params[1] - conditioner,)

ExponentialFamily.unpack_parameters(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, η) = (η[1],)
ExponentialFamily.unpack_parameters(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, η, _) = (η[1],)
