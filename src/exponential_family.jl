export ExponentialFamilyDistribution

export ExponentialFamilyDistribution, ExponentialFamilyDistributionAttributes, getnaturalparameters, getattributes
export MeanToNatural, NaturalToMean, MeanParametersSpace, NaturalParametersSpace
export getbasemeasure,
    getlogbasemeasure, getsufficientstatistics, getlogpartition, getgradlogpartition, getfisherinformation, getsupport, getmapping, getconditioner
export basemeasure, logbasemeasure, sufficientstatistics, logpartition, gradlogpartition, fisherinformation, insupport, isproper
export isbasemeasureconstant, ConstantBaseMeasure, NonConstantBaseMeasure

using LoopVectorization
using Distributions, LinearAlgebra, StaticArrays, Random

import Base: map

"""
    MeanToNatural(::Type{T})
 
Return the transformation function that maps the parameters in the mean parameters space to the natural parameters space for a distribution of type `T`.
The transformation function is of signature `(params_in_mean_space, [ conditioner ]) -> params_in_natural_space`.

See also: [`NaturalToMean`](@ref), [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref), [`getmapping`](@ref)
"""
struct MeanToNatural{T} end

MeanToNatural(::Type{T}) where {T} = MeanToNatural{T}()

# If the `conditioner` is nothing simply ignore it in the signature then
function (transformation::MeanToNatural)(params, ::Nothing)
    return transformation(params)
end

"""
    NaturalToMean(::Type{T})

Return the transformation function that maps the parameters in the natural parameters space to the mean parameters space for a distribution of type `T`.
The transformation function is of signature `(params_in_natural_space, [ conditioner ]) -> params_in_mean_space`.

See also: [`MeanToNatural`](@ref), [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref), [`getmapping`](@ref)
"""
struct NaturalToMean{T} end

NaturalToMean(::Type{T}) where {T} = NaturalToMean{T}()

# If the `conditioner` is nothing simply ignore it in the signature then
function (transformation::NaturalToMean)(params, ::Nothing)
    return transformation(params)
end

"""
    MeanParametersSpace

Specifies the mean parameters space `θ` as the desired parameters space.
Some functions (such as `logpartition` or `fisherinformation`) accept an additional `space` parameter to disambiguate the desired parameters space. 
Use `map(MeanParametersSpace() => NaturalParametersSpace(), T, parameters, conditioner)` to map the `parameters` and the `conditioner` of a distribution of type `T`
from the mean parametrization to the corresponding natural parametrization.

See also: [`NaturalParametersSpace`](@ref), [`getmapping`](@ref), [`NaturalToMean`](@ref), [`MeanToNatural`](@ref)
"""
struct MeanParametersSpace end

"""
    NaturalParametersSpace

Specifies the natural parameters space `η` as the desired parameters space.
Some functions (such as `logpartition` or `fisherinformation`) accept an additional `space` parameter to disambiguate the desired parameters space. 
Use `map(NaturalParametersSpace() => MeanParametersSpace(), T, parameters, conditioner)` to map the `parameters` and the `conditioner` of a distribution of type `T`
from the natural parametrization to the corresponding mean parametrization.

See also: [`MeanParametersSpace`](@ref), [`getmapping`](@ref), [`NaturalToMean`](@ref), [`MeanToNatural`](@ref)
"""
struct NaturalParametersSpace end

"""
    getmapping(::Pair{L, R}, T)

Returns a transformation `L -> R` between different parametrizations of a distribution of type `T`.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref), [`NaturalToMean`](@ref), [`MeanToNatural`](@ref)
"""
function getmapping end

getmapping(::Pair{NaturalParametersSpace, MeanParametersSpace}, ::Type{T}) where {T} = NaturalToMean{T}()
getmapping(::Pair{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}) where {T} = MeanToNatural{T}()

Base.map(::Pair{NaturalParametersSpace, MeanParametersSpace}, ::Type{T}, something, conditioner = nothing) where {T} =
    NaturalToMean{T}()(something, conditioner)
Base.map(::Pair{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}, something, conditioner = nothing) where {T} =
    MeanToNatural{T}()(something, conditioner)

"""
    getbasemeasure(::ExponentialFamilyDistribution)
    getbasemeasure(::Type{ <: Distribution }, [ conditioner ])

Returns the base measure function of the exponential family distribution.
"""
function getbasemeasure end

"""
    getsufficientstatistics(::ExponentialFamilyDistribution)
    getsufficientstatistics(::Type{ <: Distribution }, [ conditioner ])

Returns the list of sufficient statistics of the exponential family distribution.
"""
function getsufficientstatistics end

"""
    getlogpartition(::ExponentialFamilyDistribution)
    getlogpartition([ space ], ::Type{ <: Distribution }, [ conditioner ])

Returns the log partition function of the exponential family distribution.
"""
function getlogpartition end

"""
    getfisherinformation(::ExponentialFamilyDistribution)
    getfisherinformation([ space ], ::Type{ <: Distribution }, [ conditioner ])

Returns the function that computes the fisher information matrix of the exponential family distribution.
"""
function getfisherinformation end

"""
    getsupport(distribution_or_type)

Returns the support of the exponential family distribution.
"""
function getsupport end

"""
    ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, support)

A structure to represent the attributes of an exponential family member.

# Fields
- `basemeasure::B`: The basemeasure of the exponential family member.
- `sufficientstatistics::S`: The sufficient statistics of the exponential family member.
- `logpartition::L`: The log-partition (cumulant) of the exponential family member.
- `support::P`: The support of the exponential family member.

# Optionally
- `logbasemeasure::LB`: The log of the  basemeasure of the exponential family member.

See also: [`ExponentialFamilyDistribution`](@ref), [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getlogpartition`](@ref), [`getsupport`](@ref)
"""
struct ExponentialFamilyDistributionAttributes{B, S, L, P, LB}
    basemeasure::B
    sufficientstatistics::S
    logpartition::L
    support::P
    logbasemeasure::LB
end
function ExponentialFamilyDistributionAttributes(basemeasure::B, sufficientstatistics::S, logpartition::L, support::P) where {B, S, L, P}
    logbasemeasure = (x) -> log(basemeasure(x))
    ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, support, logbasemeasure)
end

getbasemeasure(attributes::ExponentialFamilyDistributionAttributes) = attributes.basemeasure
getsufficientstatistics(attributes::ExponentialFamilyDistributionAttributes) = attributes.sufficientstatistics
getlogpartition(attributes::ExponentialFamilyDistributionAttributes) = attributes.logpartition
getsupport(attributes::ExponentialFamilyDistributionAttributes) = attributes.support
getlogbasemeasure(attributes::ExponentialFamilyDistributionAttributes) = attributes.logbasemeasure

BayesBase.insupport(attributes::ExponentialFamilyDistributionAttributes, value) = Base.in(value, getsupport(attributes))
BayesBase.value_support(::Type{ExponentialFamilyDistributionAttributes{B, S, L, P}}) where {B, S, L, P} = value_support(P)

"""
    ExponentialFamilyDistribution(::Type{Tvar}, ::Type{Tval}, naturalparameters, conditioner, attributes)

`ExponentialFamilyDistribution` structure represents a generic exponential family distribution in natural parameterization.
Type `Tvar` can be either a distribution type (e.g. from the `Distributions.jl` package) or a variate type (e.g. `Univariate`).
Type `Tval` is can be `Continuous` or `Discrete` of type `Distributions.ValueSupport`. 
In the context of the package, exponential family distributions are represented in the form:

```math
pₓ(x ∣ η) = h(x) ⋅ exp[ η ⋅ T(x) - A(η) ]
```

Here:
- `h(x)` is the base measure.
- `T(x)` represents sufficient statistics.
- `A(η)` stands for the log partition.
- `η` denotes the natural parameters.

For a given member of exponential family: 

- `getattributes` returns either `nothing` or `ExponentialFamilyDistributionAttributes`.
- `getbasemeasure` returns a positive a valued function. 
- `getsufficientstatistics` returns an iterable of functions such as [x, x^2] or [x, logx].
- `getnaturalparameters` returns an iterable holding the values of the natural parameters. 
- `getlogpartition` return a function that depends on the naturalparameters and it ensures that the distribution is normalized to 1. 
- `getsupport` returns the set that the distribution is defined over. Could be real numbers, positive integers, 3d cube etc. Use ither the `∈` operator or the `insupport()` function to check if a value belongs to the support.

!!! note
    The `attributes` can be `nothing`. In which case the package will try to derive the corresponding attributes from the type `T`.

```jldoctest
julia> ef = convert(ExponentialFamilyDistribution, Bernoulli(0.5))
ExponentialFamily(Bernoulli, Discrete)

julia> getsufficientstatistics(ef)
(identity,)
```

```jldoctest
julia> ef = convert(ExponentialFamilyDistribution, Laplace(1.0, 0.5))
ExponentialFamily(Laplace, Continuous, conditioned on 1.0)

julia> logpdf(ef, 4.0)
-6.0
```

See also: [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getnaturalparameters`](@ref), [`getlogpartition`](@ref), [`getsupport`](@ref)
"""
struct ExponentialFamilyDistribution{Tvar, Tval, P, C, A}
    naturalparameters::P
    conditioner::C
    attributes::A

    ExponentialFamilyDistribution(
        ::Type{Tvar},
        ::Type{Tval},
        naturalparameters::P,
        conditioner::C = nothing,
        attributes::A = nothing
    ) where {Tvar, Tval, P, C, A} = begin
        new{Tvar, Tval, P, C, A}(
            naturalparameters,
            conditioner,
            attributes
        )
    end
end

function ExponentialFamilyDistribution(
    ::Type{Tvar},
    naturalparameters::P,
    conditioner = nothing
) where {Tvar <: Distribution, P}
    if !isproper(NaturalParametersSpace(), Tvar, naturalparameters, conditioner)
        error(lazy"Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T).")
    end
    return ExponentialFamilyDistribution(Tvar, value_support(Tvar), naturalparameters, conditioner, nothing)
end

function Base.show(io::IO, ef::ExponentialFamilyDistribution{Tvar, Tval}) where {Tvar, Tval}
    print(io, "ExponentialFamily(", Tvar,", ", Tval)
    conditioner = getconditioner(ef)
    if !isnothing(conditioner)
        print(io, ", conditioned on ", conditioner)
    end
    print(io, ")")
end

"""
    isproper(::ExponentialFamilyDistribution)

Checks if the object of type `ExponentialFamilyDistribution` is a proper distribution.
"""
isproper(::ExponentialFamilyDistribution) = true

"""
    getnaturalparameters(::ExponentialFamilyDistribution)

Get the natural parameters of the exponential family distribution.
"""
getnaturalparameters(ef::ExponentialFamilyDistribution) = ef.naturalparameters

"""
    getconditioner(::ExponentialFamilyDistribution)

Returns either the conditioner of the exponential family distribution or `nothing`. `conditioner` is a fixed parameter that is used to ensure that the distribution belongs to the exponential family.
"""
getconditioner(ef::ExponentialFamilyDistribution) = ef.conditioner

"""
    getattributes(::ExponentialFamilyDistribution)

Returns iether the attributes of the exponential family member or `nothing`. 

See also: [`ExponentialFamilyDistributionAttributes`](@ref)
"""
getattributes(ef::ExponentialFamilyDistribution) = ef.attributes

"""
    basemeasure(::ExponentialFamilyDistribution, x)

Returns the computed value of `basemeasure` of the exponential family distribution at the point `x`.

See also: [`getbasemeasure`](@ref)
"""
function basemeasure(ef::ExponentialFamilyDistribution, x)
    return getbasemeasure(ef)(x)
end

"""
    logbasemeasure(::ExponentialFamilyDistribution, x)

Returns the computed value of `log(basemeasure)` of the exponential family distribution at the point `x`.

See also [`basemeasure`](@ref)
"""
function logbasemeasure(ef::ExponentialFamilyDistribution, x)
    return getlogbasemeasure(ef)(x)
end

"""
    sufficientstatistics(::ExponentialFamilyDistribution)

Returns the computed values of `sufficientstatistics` of the exponential family distribution at the point `x`.
"""
function sufficientstatistics(ef::ExponentialFamilyDistribution, x)
    return map(f -> f(x), getsufficientstatistics(ef))
end

"""
    logpartition(::ExponentialFamilyDistribution, η)

Return the computed value of `logpartition` of the exponential family distribution at the point `η`.
By default `η = getnaturalparameters(ef)`.

See also: [`getlogpartition`](@ref)
"""
function logpartition(ef::ExponentialFamilyDistribution, η = getnaturalparameters(ef))
    return getlogpartition(ef)(η)
end

"""
    gradlogpartition(::ExponentialFamilyDistribution, η)

Return the computed value of `gradlogpartition` of the exponential family distribution at the point `η`.
By default `η = getnaturalparameters(ef)`.

See also: [`getgradlogpartition`](@ref)
"""
function gradlogpartition(ef::ExponentialFamilyDistribution, η = getnaturalparameters(ef))
    return getgradlogpartition(ef)(η)
end

"""
    fisherinformation(distribution, η)

Return the computed value of `fisherinformation` of the exponential family distribution at the point `η`
By default `η = getnaturalparameters(ef)`.

See also: [`getfisherinformation`](@ref)
"""
function fisherinformation(ef::ExponentialFamilyDistribution, η = getnaturalparameters(ef))
    return getfisherinformation(ef)(η)
end

getbasemeasure(ef::ExponentialFamilyDistribution) = getbasemeasure(ef.attributes, ef)
getbasemeasure(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getbasemeasure(T, getconditioner(ef))
getbasemeasure(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getbasemeasure(attributes)

getlogbasemeasure(ef::ExponentialFamilyDistribution) = getlogbasemeasure(ef.attributes, ef)
getlogbasemeasure(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getlogbasemeasure(T, getconditioner(ef))
getlogbasemeasure(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getlogbasemeasure(attributes)

getsufficientstatistics(ef::ExponentialFamilyDistribution) = getsufficientstatistics(ef.attributes, ef)
getsufficientstatistics(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} =
    getsufficientstatistics(T, getconditioner(ef))
getsufficientstatistics(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getsufficientstatistics(attributes)

getlogpartition(ef::ExponentialFamilyDistribution) = getlogpartition(ef.attributes, ef)
getlogpartition(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getlogpartition(T, getconditioner(ef))
getlogpartition(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getlogpartition(attributes)

getgradlogpartition(ef::ExponentialFamilyDistribution) = getgradlogpartition(ef.attributes, ef)
getgradlogpartition(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} =
    getgradlogpartition(T, getconditioner(ef))
# TODO: Implement Monte Carlo estimation for gradlogpartition.
getgradlogpartition(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    error("Generic gradlogpartition is not implemented.")

getfisherinformation(ef::ExponentialFamilyDistribution) = getfisherinformation(ef.attributes, ef)
getfisherinformation(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} =
    getfisherinformation(T, getconditioner(ef))
# TODO: Implement a generic fisherinformation.
getfisherinformation(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    error("Generic getfisherinformation is not implemented.")

getsupport(ef::ExponentialFamilyDistribution) = getsupport(ef.attributes, ef)
getsupport(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getsupport(T)
getsupport(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getsupport(attributes)

BayesBase.insupport(ef::ExponentialFamilyDistribution, value) = Base.in(value, getsupport(ef))

# For all `<:Distribution` the `support` function should be defined
getsupport(::Type{T}) where {T <: Distribution} = BayesBase.support(T)

# Convenient mappings from a vectorized form to a vectorized form for distributions
function (transformation::NaturalToMean{T})(v::AbstractVector) where {T <: Distribution}
    return transformation(v, nothing)
end

function (transformation::NaturalToMean{T})(v::AbstractVector, ::Nothing) where {T <: Distribution}
    return pack_parameters(MeanParametersSpace(), T, transformation(unpack_parameters(NaturalParametersSpace(), T, v)))
end

function (transformation::NaturalToMean{T})(v::AbstractVector, conditioner) where {T <: Distribution}
    return pack_parameters(MeanParametersSpace(), T, transformation(unpack_parameters(NaturalParametersSpace(), T, v, conditioner), conditioner))
end

function (transformation::MeanToNatural{T})(v::AbstractVector) where {T <: Distribution}
    return transformation(v, nothing)
end

function (transformation::MeanToNatural{T})(v::AbstractVector, ::Nothing) where {T <: Distribution}
    return pack_parameters(NaturalParametersSpace(), T, transformation(unpack_parameters(MeanParametersSpace(), T, v)))
end

function (transformation::MeanToNatural{T})(v::AbstractVector, conditioner) where {T <: Distribution}
    return pack_parameters(NaturalParametersSpace(), T, transformation(unpack_parameters(MeanParametersSpace(), T, v, conditioner), conditioner))
end

"""
    isproper([ space = NaturalParametersSpace() ], ::Type{T}, parameters, conditioner = nothing) where { T <: Distribution }

A specific verion of `isproper` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.
For conditional exponential family distributions requires an extra `conditioner` argument.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
isproper(::Type{T}, parameters, conditioner = nothing) where {T <: Distribution} =
    isproper(NaturalParametersSpace(), T, parameters, conditioner)
isproper(space::Union{NaturalParametersSpace, MeanParametersSpace}, ::Type{T}, parameters) where {T} =
    isproper(space, T, parameters, nothing)
isproper(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} =
    isproper(NaturalParametersSpace(), T, getnaturalparameters(ef), getconditioner(ef))

"""
    getbasemeasure(::Type{<:Distribution}, [ conditioner ])

A specific verion of `getbasemeasure` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
For conditional exponential family distributions requires an extra `conditioner` argument.
"""
getbasemeasure(::Type{T}, ::Nothing) where {T <: Distribution} = getbasemeasure(T)

"""
    getlogbasemeasure(::Type{<:Distribution}, [ conditioner ])

A generic verion of `getlogbasemeasure` defined particularly for distribution types from `Distributions.jl` package.
Just computes log of basemeasure.
"""
getlogbasemeasure(::Type{T}) where {T <: Distribution} = (x) -> log(getbasemeasure(T)(x))

"""
    getlogbasemeasure(::Type{<:Distribution}, [ conditioner ])

A generic verion of `getbasemeasure` defined particularly for distribution types from `Distributions.jl` package.
For conditional exponential family distributions requires an extra `conditioner` argument. Just computes log of basemeasure.
"""
getlogbasemeasure(::Type{T}, ::Nothing) where {T <: Distribution} = getlogbasemeasure(T)

"""
    getsufficientstatistics(::Type{<:Distribution}, [ conditioner ])

A specific verion of `getsufficientstatistics` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
For conditional exponential family distributions requires an extra `conditioner` argument.
"""
getsufficientstatistics(::Type{T}, ::Nothing) where {T <: Distribution} = getsufficientstatistics(T)

"""
    getlogpartition([ space = NaturalParametersSpace() ], ::Type{T}, [ conditioner ]) where { T <: Distribution }

A specific verion of `getlogpartition` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.
For conditional exponential family distributions requires an extra `conditioner` argument.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
getlogpartition(::Type{T}, conditioner = nothing) where {T <: Distribution} =
    getlogpartition(NaturalParametersSpace(), T, conditioner)
getlogpartition(
    space::Union{MeanParametersSpace, NaturalParametersSpace},
    ::Type{T},
    ::Nothing
) where {T <: Distribution} = getlogpartition(space, T)

"""
    getgradlogpartition([ space = NaturalParametersSpace() ], ::Type{T}, [ conditioner ]) where { T <: Distribution }

A specific verion of `getgradlogpartition` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.
For conditional exponential family distributions requires an extra `conditioner` argument.
"""
getgradlogpartition(::Type{T}, conditioner = nothing) where {T <: Distribution} =
    getgradlogpartition(NaturalParametersSpace(), T, conditioner)
getgradlogpartition(
    space::Union{MeanParametersSpace, NaturalParametersSpace},
    ::Type{T},
    ::Nothing
) where {T <: Distribution} = getgradlogpartition(space, T)

"""
    getfisherinformation([ space = NaturalParametersSpace() ], ::Type{T}) where { T <: Distribution }

A specific verion of `getfisherinformation` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.
For conditional exponential family distributions requires an extra `conditioner` argument.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
getfisherinformation(::Type{T}, conditioner = nothing) where {T <: Distribution} =
    getfisherinformation(NaturalParametersSpace(), T, conditioner)
getfisherinformation(
    space::Union{MeanParametersSpace, NaturalParametersSpace},
    ::Type{T},
    ::Nothing
) where {T <: Distribution} = getfisherinformation(space, T)

"""
A trait object representing that the base measure is constant.
"""
struct ConstantBaseMeasure end

"""
A trait object representing that the base measure is **not** constant.
"""
struct NonConstantBaseMeasure end

"""
    isbasemeasureconstant(something)

Returns either `NonConstantBaseMeasure()` or `ConstantBaseMeasure()` depending on if the base measure is a constant with respect to the natural parameters of `something` or not.
By default the package assumes that any base measure in a form of the `Function` is not a constant.
It, however, is not true for basemeasure that simply return a constant. In such cases the `isbasemeasureconstant` must have a specific method.

See also: [`getbasemeasure`](@ref), [`basemeasure`](@ref)
"""
function isbasemeasureconstant end

isbasemeasureconstant(ef::ExponentialFamilyDistribution) = isbasemeasureconstant(getbasemeasure(ef))
isbasemeasureconstant(::ExponentialFamilyDistribution{T}) where {T <: Distribution} = isbasemeasureconstant(T)

# For safety, by default, we assume that any base measure function is not a constant
# This is not always the case, e.g. when a function is (η) -> 1
# In such cases the `isbasemeasureconstant` must have a specific method
isbasemeasureconstant(::Function) = NonConstantBaseMeasure()

"""
    logpdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the log-density of the exponential family distribution for the input `x`.
"""
function BayesBase.logpdf(ef::ExponentialFamilyDistribution, x)
    return _logpdf(ef, x)
end

"""
A trait object, signifying that the _logpdf method should treat it second argument as one point from the distrubution domain.
"""
struct PointBasedLogpdfCall end

"""
A trait object, signifying that the _logpdf method should treat it second argument as a container of points from the distrubution domain.
"""
struct MapBasedLogpdfCall end

function _logpdf(::PointBasedLogpdfCall, ef, x)
    _plogpdf(ef, x)
end

function _logpdf(::MapBasedLogpdfCall, ef, container)
    _vlogpdf(ef, container)
end

"""
    _logpdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the log-density of the exponential family distribution for the input `x`.

This inner function dispatches to the appropriate version of `_logpdf` based on the types of `x` and `ef`, utilizing the `check_logpdf` function. The dispatch mechanism ensures that `_logpdf` correctly handles the input `x`, whether it is a single point or a container of points, according to the nature of the exponential family distribution and `x`.

For instance, with a `Univariate` distribution, `_logpdf` evaluates the log-density for a single point if `x` is a `Number`, and for a container of points if `x` is an `AbstractVector`.

### Examples
Evaluate the log-density of a Gamma distribution at a single point:

```jldoctest
using ExponentialFamily, Distributions;
gamma = convert(ExponentialFamilyDistribution, Gamma(1, 1))
ExponentialFamily._logpdf(gamma, 1.0)
# output
-1.0
```

Evaluate the log-density of a Gamma distribution at multiple points:

```jldoctest
using ExponentialFamily, Distributions
gamma = convert(ExponentialFamilyDistribution, Gamma(1, 1))
ExponentialFamily._logpdf(gamma, [1, 2, 3])
# output
3-element Vector{Float64}:
 -1.0
 -2.0
 -3.0
```

For details on the dispatch mechanism of `_logpdf`, refer to the `check_logpdf` function.

See also: [`check_logpdf`](@ref)
"""
function _logpdf(ef::ExponentialFamilyDistribution{T}, x) where {T}
    vartype, _x = check_logpdf(ef, x)
    _logpdf(vartype, ef, _x)
end

function _plogpdf(ef, x)
    @assert insupport(ef, x) lazy"Point $(x) does not belong to the support of $(ef)"
    return _plogpdf(ef, x, logpartition(ef), logbasemeasure(ef, x))
end

_scalarproduct(::Type{T}, η, statistics) where {T} = _scalarproduct(variate_form(T), T, η, statistics)
_scalarproduct(::Type{Univariate}, η, statistics) = dot(η, flatten_parameters(statistics))
_scalarproduct(::Type{Univariate}, ::Type{T}, η, statistics) where {T} = dot(η, flatten_parameters(T, statistics))
_scalarproduct(_, ::Type{T}, η, statistics) where {T} = dot(η, pack_parameters(T, statistics))

function _plogpdf(ef::ExponentialFamilyDistribution{T}, x, logpartition, logbasemeasure) where {T}
    # TODO: Think of what to do with this assert
    @assert insupport(ef, x) lazy"Point $(x) does not belong to the support of $(ef)"
    η = getnaturalparameters(ef)
    _statistics = sufficientstatistics(ef, x)
    return logbasemeasure + _scalarproduct(T, η, _statistics) - logpartition
end

"""
    check_logpdf(variate_form, typeof(x), eltype(x), ef, x)

Determines an appropriate strategy of evaluation of `_logpdf` (`PointBasedLogpdfCall` or `MapBasedLogpdfCall`) to use based on the types of `x` and `ef`. This function employs a dispatch mechanism that adapts to the input `x`, whether it is a single point or a container of points, in accordance with the characteristics of the exponential family distribution (`ef`) and the variate form of `x`.

### Strategies
- For a `Univariate` distribution:
  - If `x` is a `Number`, `_logpdf` is invoked with `PointBasedLogpdfCall()`.
  - If `x` is an `AbstractVector` containing `Number`s, `_logpdf` is invoked with `MapBasedLogpdfCall()`.

- For a `Multivariate` distribution:
  - If `x` is an `AbstractVector` containing `Number`s, `_logpdf` is invoked with `PointBasedLogpdfCall()`.
  - If `x` is an `AbstractVector` containing `AbstractVector`s, `_logpdf` is invoked with `MapBasedLogpdfCall()`.
  - If `x` is an `AbstractMatrix` containing `Number`s, `_logpdf` is invoked with `MapBasedLogpdfCall()`, transforming `x` to `eachcol(x)`.

- For a `Matrixvariate` distribution:
  - If `x` is an `AbstractMatrix` containing `Number`s, `_logpdf` is invoked with `PointBasedLogpdfCall()`.
  - If `x` is an `AbstractVector` containing `AbstractMatrix`s, `_logpdf` is invoked with `MapBasedLogpdfCall()`.

### Examples
```jldoctest
using ExponentialFamily
ExponentialFamily.check_logpdf(Univariate, typeof(1.0), eltype(1.0), Gamma(1, 1), 1.0)
# output
(ExponentialFamily.PointBasedLogpdfCall(), 1.0)
```

```jldoctest
using ExponentialFamily
ExponentialFamily.check_logpdf(Univariate, typeof([1.0, 2.0, 3.0]), eltype([1.0, 2.0, 3.0]), Gamma(1, 1), [1.0, 2.0, 3.0])
# output
(ExponentialFamily.MapBasedLogpdfCall(), [1.0, 2.0, 3.0])
```

See also: [`_logpdf`](@ref) [`PointBasedLogpdfCall`](@ref) [`MapBasedLogpdfCall`](@ref)
"""
function check_logpdf end

check_logpdf(ef::ExponentialFamilyDistribution, x) = check_logpdf(variate_form(typeof(ef)), typeof(x), eltype(x), ef, x)

check_logpdf(::Type{Univariate}, ::Type{<:Number}, ::Type{<:Number}, ef, x) = (PointBasedLogpdfCall(), x)
check_logpdf(::Type{Multivariate}, ::Type{<:AbstractVector}, ::Type{<:Number}, ef, x) = (PointBasedLogpdfCall(), x)
check_logpdf(::Type{Matrixvariate}, ::Type{<:AbstractMatrix}, ::Type{<:Number}, ef, x) = (PointBasedLogpdfCall(), x)
check_logpdf(::Type{VectorMatrixvariate},  ::Type{<:AbstractVector}, ::Type{<:Tuple}, ef, x) = (PointBasedLogpdfCall(), x)

function _vlogpdf(ef, container)
    _logpartition = logpartition(ef)
    return map(x -> _plogpdf(ef, x, _logpartition, logbasemeasure(ef, x)), container)
end

check_logpdf(::Type{Univariate}, ::Type{<:AbstractVector}, ::Type{<:Number}, ef, container) = (MapBasedLogpdfCall(), container)
check_logpdf(::Type{Multivariate}, ::Type{<:AbstractVector}, ::Type{<:AbstractVector}, ef, container) = (MapBasedLogpdfCall(), container)
check_logpdf(::Type{Multivariate}, ::Type{<:AbstractMatrix}, ::Type{<:Number}, ef, container) = (MapBasedLogpdfCall(), eachcol(container))
check_logpdf(::Type{Matrixvariate}, ::Type{<:AbstractVector}, ::Type{<:AbstractMatrix}, ef, container) = (MapBasedLogpdfCall(), container)
"""
    pdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the probability density function of the exponential family distribution for the input `x`.
"""
BayesBase.pdf(ef::ExponentialFamilyDistribution, x) = _pdf(ef, x)

function _pdf(ef, x)
    vartype, _x = check_logpdf(ef, x)
    _pdf(vartype, ef, _x)
end

function _pdf(::PointBasedLogpdfCall, ef, x)
    exp(logpdf(ef, x))
end

function _pdf(::MapBasedLogpdfCall, ef, x)
    exp.(logpdf(ef, x))
end

"""
    cdf(ef::ExponentialFamilyDistribution{D}, x) where { D <: Distribution }

Evaluates and returns the cumulative distribution function of the exponential family distribution for the input `x`.
"""
BayesBase.cdf(ef::ExponentialFamilyDistribution{D}, x) where {D <: Distribution} = cdf(Base.convert(Distribution, ef), x)

BayesBase.variate_form(::Type{<:ExponentialFamilyDistribution{D}}) where {D <: Distribution} = variate_form(D)
BayesBase.variate_form(::Type{<:ExponentialFamilyDistribution{V}}) where {V <: VariateForm} = V

BayesBase.value_support(::Type{<:ExponentialFamilyDistribution{D}}) where {D <: Distribution} = value_support(D)
BayesBase.value_support(::Type{<:ExponentialFamilyDistribution{D, P, C, A}}) where {D, P, C, A} = value_support(A)

"""
    flatten_parameters(::Type{T}, params::Tuple)

This function returns the parameters of a distribution of type `T` in a flattened form without actually allocating the container.
"""
flatten_parameters(params::Tuple) = Iterators.flatten(params)

# Ignore the `T` by default, assume all the distribution can be flattened with the generic version 
# If the assumption does not hold, implement a specific method
flatten_parameters(::Type{T}, params::Tuple) where {T} = flatten_parameters(params)

"""
    pack_parameters([ space ], ::Type{T}, params::Tuple)

This function returns the parameters of a distribution of type `T` in a vectorized (packed) form. For most of the distributions the packed versions are of the 
same structure in any parameters space. For some distributions, however, it is necessary to indicate the `space` of the packaged parameters.

```jldoctest
julia> ExponentialFamily.pack_parameters((1, [2.0, 3.0], [4.0 5.0 6.0; 7.0 8.0 9.0]))
9-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
 7.0
 5.0
 8.0
 6.0
 9.0
```
"""
function pack_parameters end

# Assume that for the most distributions the `pack_parameters` does not depend on the `space` parameter
pack_parameters(::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}, params::Tuple) where {T} = pack_parameters(T, params)
pack_parameters(::Type{T}, params::Tuple) where {T <: Distribution} = pack_parameters(params)

# Below is an optimized version of packing, which assumes that the packed container is 
# an array, what it does essentially is 
# 1. preallocates the `container` of the needed length
# 2. recursively iterates through the tuple of parameters 
# 3. for each parameter copies the content into the preallocated container without checking
function pack_parameters(params::Tuple)
    lengths = map(length, params)
    len = sum(lengths)
    F = mapreduce(ExponentialFamily.deep_eltype, promote_type, params)
    container = Vector{F}(undef, len)
    return __pack_parameters_fast!(container, 1, 1, lengths, Base.first(params), Base.tail(params))
end

function __pack_parameters_fast!(container::Vector, offset::Int, current::Int, lengths, front, tail::Tuple)
    N = lengths[current]
    __pack_copyto!(container, offset, front, N)
    return __pack_parameters_fast!(container, offset + N, current + 1, lengths, Base.first(tail), Base.tail(tail))
end

function __pack_parameters_fast!(container::Vector, i::Int, k::Int, lengths, front, ::Tuple{})
    N = lengths[k]
    __pack_copyto!(container, i, front, N)
    return container
end

__pack_copyto!(dest, doffset, source, n) = copyto!(dest, doffset, source, firstindex(source), n)
__pack_copyto!(dest::Array, doffset, source::Array, n) = unsafe_copyto!(dest, doffset, source, firstindex(source), n)
__pack_copyto!(dest::Array, doffset, source::Number, _) = @inbounds(dest[doffset] = source)

"""
    unpack_parameters([ space ], ::Type{T}, parameters)

This function "unpack" the vectorized form of the parameters in a tuple. For most of the distributions the packed `parameters` are of the 
same structure in any parameters space. For some distributions, however, it is necessary to indicate the `space` of the packaged parameters.

See also: [`MeanParametersSpace`](@ref), [`NaturalParametersSpace`](@ref)
"""
function unpack_parameters end

unpack_parameters(ef::ExponentialFamilyDistribution{T}) where {T} = 
    unpack_parameters(NaturalParametersSpace(), T, getnaturalparameters(ef), getconditioner(ef))

function unpack_parameters(::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}, packed, conditioner) where {T}
    unpack_parameters(T, packed, conditioner)
end

function unpack_parameters(::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}, packed, ::Nothing) where {T}
    unpack_parameters(T, packed)
end

# Assume that for the most distributions the `unpack_parameters` does not depend on the `space` parameter
unpack_parameters(::Union{MeanParametersSpace, NaturalParametersSpace}, ::Type{T}, packed) where {T} = unpack_parameters(T, packed)

"""
    separate_conditioner(::Type{T}, params) where {T <: Distribution}

Separates the conditioner argument from `params` and returns a tuple of `(conditioned_params, conditioner)`.
By default returns `(params, nothing)` but can be overwritten for certain distributions.

```jldoctest
julia> (cparams, conditioner) = ExponentialFamily.separate_conditioner(Laplace, (0.0, 1.0))
((1.0,), 0.0)

julia> params = ExponentialFamily.join_conditioner(Laplace, cparams, conditioner)
(0.0, 1.0)

julia> Laplace(params...) == Laplace(0.0, 1.0)
true
```

See also: [`ExponentialFamily.join_conditioner`](@ref)
"""
separate_conditioner(::Type{T}, params) where {T <: Distribution} = (params, nothing)

"""
    join_conditioner(::Type{T}, params, conditioner) where { T <: Distribution }

Joins the conditioner argument with the `params` and returns a tuple of joined params, such that it can be used in a constructor of the `T` distribution.

```jldoctest
julia> (cparams, conditioner) = ExponentialFamily.separate_conditioner(Laplace, (0.0, 1.0))
((1.0,), 0.0)

julia> params = ExponentialFamily.join_conditioner(Laplace, cparams, conditioner)
(0.0, 1.0)

julia> Laplace(params...) == Laplace(0.0, 1.0)
true
```

See also: [`ExponentialFamily.separate_conditioner`](@ref)
"""
join_conditioner(::Type{T}, params, ::Nothing) where {T <: Distribution} = params

Base.convert(::Type{T}, ef::ExponentialFamilyDistribution{E}) where {T <: Distribution, E <: Distribution} =
    Base.convert(T, Base.convert(Distribution, ef))

# Assume that the type tag is the same as the `Distribution` type but without the type parameters 
# This can be overwritten by certain distributions, which have many different parametrizations, e.g. `Gamma` or `Normal`
# The package also makes the assumption that the `MeanParametersSpace` **is** the of same type as `exponential_family_typetag`
exponential_family_typetag(distribution) = distribution_typewrapper(distribution)
exponential_family_typetag(::ExponentialFamilyDistribution{D}) where {D} = D

BayesBase.params(::MeanParametersSpace, distribution::Distribution) = params(distribution)

function BayesBase.params(::NaturalParametersSpace, distribution::Distribution)
    θ = params(MeanParametersSpace(), distribution)
    return map(MeanParametersSpace() => NaturalParametersSpace(), exponential_family_typetag(distribution), θ)
end

BayesBase.value_support(::ExponentialFamilyDistribution{Tvar, Tval}) where {Tvar, Tval} = Tval

Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{T}) where {T} =
    error("Cannot convert an arbitrary `ExponentialFamily{$T}` object to a `Distribution`. An explicit approximation method is required.")

# Generic convert from an `ExponentialFamilyDistribution{T}` to its corresponding type `T`
function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{T}) where {T <: Distribution}
    tuple_of_η = unpack_parameters(ef)
    conditioner = getconditioner(ef)
    # Map the conditioned natural parameters space into its corresponding mean parameters space
    cparams = map(NaturalParametersSpace() => MeanParametersSpace(), T, tuple_of_η, conditioner)
    # `Distributions.jl` stores the params in a single tuple, so we need to join the parameters and the conditioner
    params = join_conditioner(T, cparams, conditioner)
    return T(params...)
end

# Generic convert from a member of `Distributions` to its equivalent representation in `ExponentialFamilyDistribution`
function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Distribution)
    # Get the type wrapper, e.g. `Bernoulli{Float64, ...}` becomes just `Bernoulli`
    T = exponential_family_typetag(dist)

    tuple_of_θ = params(MeanParametersSpace(), dist)
    # Separate the parameters and the conditioner, the `params` function returns all together
    cparams, conditioner = separate_conditioner(T, tuple_of_θ)

    # Map the conditioned `cparams` into the natural parameters space
    tuple_of_η = map(MeanParametersSpace() => NaturalParametersSpace(), T, cparams, conditioner)
    # Pack the parameters for efficiency
    η = pack_parameters(NaturalParametersSpace(), T, tuple_of_η)

    return ExponentialFamilyDistribution(T, η, conditioner)
end

function BayesBase.paramfloattype(ef::ExponentialFamilyDistribution)
    return deep_eltype(getnaturalparameters(ef))
end

function BayesBase.convert_paramfloattype(::Type{F}, ef::ExponentialFamilyDistribution{Tvar, Tval}) where {F, Tvar, Tval}
    return ExponentialFamilyDistribution(
        Tvar,
        Tval,
        convert_paramfloattype(F, getnaturalparameters(ef)),
        getconditioner(ef),
        getattributes(ef)
    )
end

BayesBase.mean(f::F, ef::ExponentialFamilyDistribution{T}) where {F, T <: Distribution} = mean(f, convert(T, ef))
BayesBase.mean(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = mean(convert(T, ef))
BayesBase.var(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = var(convert(T, ef))
BayesBase.std(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = std(convert(T, ef))
BayesBase.cov(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = cov(convert(T, ef))
BayesBase.skewness(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = skewness(convert(T, ef))
BayesBase.kurtosis(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = kurtosis(convert(T, ef))

BayesBase.rand(ef::ExponentialFamilyDistribution, args...) = rand(Random.default_rng(), ef, args...)
BayesBase.rand!(ef::ExponentialFamilyDistribution, args...) = rand!(Random.default_rng(), ef, args...)

BayesBase.rand(ef::ExponentialFamilyDistribution, args::Integer...) = rand(Random.default_rng(), ef, args...)

BayesBase.rand(rng::AbstractRNG, ef::ExponentialFamilyDistribution{T}, args::Integer...) where {T <: Distribution} =
    rand(rng, convert(T, ef), args...)

BayesBase.rand!(rng::AbstractRNG, ef::ExponentialFamilyDistribution{T}, container) where {T <: Distribution} =
    rand!(rng, convert(T, ef), container)

Base.isapprox(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution; kwargs...) = false
Base.:(==)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) = false

function Base.isapprox(
    left::ExponentialFamilyDistribution{Tvar, Tval},
    right::ExponentialFamilyDistribution{Tvar, Tval};
    kwargs...
) where {Tvar, Tval}
    return getbasemeasure(left) == getbasemeasure(right) &&
           getsufficientstatistics(left) == getsufficientstatistics(right) &&
           getlogpartition(left) == getlogpartition(right) && getsupport(left) == getsupport(right) &&
           getconditioner(left) == getconditioner(right) &&
           isapprox(getnaturalparameters(left), getnaturalparameters(right); kwargs...)
end

function Base.:(==)(left::ExponentialFamilyDistribution{Tvar, Tval}, right::ExponentialFamilyDistribution{Tvar, Tval}) where {Tvar, Tval}
    return getbasemeasure(left) == getbasemeasure(right) &&
           getsufficientstatistics(left) == getsufficientstatistics(right) &&
           getlogpartition(left) == getlogpartition(right) && getsupport(left) == getsupport(right) &&
           getconditioner(left) == getconditioner(right) &&
           getnaturalparameters(left) == getnaturalparameters(right)
end

function Base.isapprox(left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T};
    kwargs...) where {T <: Distribution}
    if (isnothing(getconditioner(left)) && isnothing(getconditioner(right)))
        return isapprox(getnaturalparameters(left), getnaturalparameters(right); kwargs...)
    end
    return isapprox(getconditioner(left), getconditioner(right); kwargs...) &&
           isapprox(getnaturalparameters(left), getnaturalparameters(right); kwargs...)
end

function Base.:(==)(left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}) where {T <: Distribution}
    if (isnothing(getconditioner(left)) && isnothing(getconditioner(right)))
        return getnaturalparameters(left) == getnaturalparameters(right)
    end
    return getconditioner(left) == getconditioner(right) && getnaturalparameters(left) == getnaturalparameters(right)
end

Base.similar(ef::ExponentialFamilyDistribution) = similar(ef, eltype(getnaturalparameters(ef)))

function Base.similar(ef::ExponentialFamilyDistribution{Tvar, Tval}, ::Type{F}) where {Tvar, Tval, F}
    return ExponentialFamilyDistribution(Tvar, Tval, similar(getnaturalparameters(ef), F), getconditioner(ef), getattributes(ef))
end

BayesBase.vague(::Type{ExponentialFamilyDistribution{T}}, args...) where {T <: Distribution} =
    convert(ExponentialFamilyDistribution, vague(T, args...))

# We assume that we want to preserve the `ExponentialFamilyDistribution` when working with two `ExponentialFamilyDistribution`s
BayesBase.default_prod_rule(::Type{<:ExponentialFamilyDistribution}, ::Type{<:ExponentialFamilyDistribution}) =
    PreserveTypeProd(ExponentialFamilyDistribution)

function BayesBase.prod(::ClosedProd, left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution)
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
end

function BayesBase.prod(::PreserveTypeProd{ExponentialFamilyDistribution}, left::Distribution, right::Distribution)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left, ef_right)
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{Tvar, Tval},
    right::ExponentialFamilyDistribution{Tvar, Tval}
) where {Tvar, Tval}
    # Se here we assume that if both left has the exact same base measure and this base measure is `ConstantBaseMeasure`
    # We assume that this code-path is static and should be const-folded in run-time (there are tests that check that this function does not allocated more than `similar(left)`)
    if isbasemeasureconstant(left) === ConstantBaseMeasure() &&
       isbasemeasureconstant(right) === ConstantBaseMeasure() && getbasemeasure(left) === getbasemeasure(right)
        # Check that both conditioners are either nothing or all are approximately equal
        if (isnothing(getconditioner(left)) && isnothing(getconditioner(right))) || (isapprox(getconditioner(left), getconditioner(right)))
            # Find the promoted float type of both natural parameters
            F = promote_type(eltype(getnaturalparameters(left)), eltype(getnaturalparameters(right)))
            # Create a suitable container for the in-place `prod!` operation, we use the `left` as a suitable candidate
            container = similar(left, F)
            return BayesBase.prod!(container, left, right)
        end
    end
    error("Generic product of two exponential family members is not implemented.")
end

function BayesBase.prod!(
    container::ExponentialFamilyDistribution{Tvar, Tval},
    left::ExponentialFamilyDistribution{Tvar, Tval},
    right::ExponentialFamilyDistribution{Tvar, Tval}
) where {Tvar, Tval}
    # First check if we can actually simply sum-up the natural parameters
    # We assume that this code-path is static and should be const-folded in run-time (there are tests that check that this function does not allocate in this simple case)
    if isbasemeasureconstant(left) === ConstantBaseMeasure() &&
       isbasemeasureconstant(right) === ConstantBaseMeasure() && getbasemeasure(left) === getbasemeasure(right)
        # Check that all three conditioners are either nothing or all are approximately equal
        if (isnothing(getconditioner(container)) && isnothing(getconditioner(left)) && isnothing(getconditioner(right))) ||
           (isapprox(getconditioner(left), getconditioner(right)) && isapprox(getconditioner(container), getconditioner(left)))
            LoopVectorization.vmap!(
                +,
                getnaturalparameters(container),
                getnaturalparameters(left),
                getnaturalparameters(right)
            )
            return container
        end
    end
    # If the check fails, do not do un-safe operation and simply fallback to the `PreserveTypeProd(ExponentialFamilyDistribution)`
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
end
