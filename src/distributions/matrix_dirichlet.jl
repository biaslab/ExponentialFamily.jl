export MatrixDirichlet

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf

import SparseArrays: blockdiag, sparse
import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra, Random

"""
    MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution

A matrix-valued MatrixDirichlet distribution, where `T` is the element type of the matrix `A`.
The `a` field stores the matrix parameter of the distribution.

# Fields
- `a::A`: The matrix parameter of the MatrixDirichlet distribution.
"""
struct MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    a::A
end

BayesBase.mean(dist::MatrixDirichlet) = dist.a ./ sum(dist.a; dims = 1)

function BayesBase.cov(dist::MatrixDirichlet)
    matrices = vmap(
        d -> sparse(cov(Dirichlet(convert(Vector, d)))),
        eachcol(dist.a)
    )

    return vmapreduce(identity, blockdiag, matrices)
end

BayesBase.var(dist::MatrixDirichlet) = diag(cov(dist))
BayesBase.std(dist::MatrixDirichlet) = vmap(sqrt, var(dist))

BayesBase.params(dist::MatrixDirichlet) = (dist.a,)

Base.size(dist::MatrixDirichlet) = size(dist.a)
Base.eltype(::MatrixDirichlet{T}) where {T} = T

BayesBase.vague(::Type{<:MatrixDirichlet}, dims::Int)              = MatrixDirichlet(ones(dims, dims))
BayesBase.vague(::Type{<:MatrixDirichlet}, dims1::Int, dims2::Int) = MatrixDirichlet(ones(dims1, dims2))
BayesBase.vague(::Type{<:MatrixDirichlet}, dims::Tuple)            = MatrixDirichlet(ones(dims))

function BayesBase.entropy(dist::MatrixDirichlet)
    return vmapreduce(+, eachcol(dist.a)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
BayesBase.promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
BayesBase.promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet

function BayesBase.rand(rng::AbstractRNG, dist::MatrixDirichlet{T}) where {T}
    container = similar(dist.a)
    return rand!(rng, dist, container)
end

function BayesBase.rand(rng::AbstractRNG, dist::MatrixDirichlet{T}, nsamples::Int64) where {T}
    container = Vector{Matrix{T}}(undef, nsamples)
    @inbounds for i in eachindex(container)
        container[i] = Matrix{T}(undef, size(dist))
        rand!(rng, dist, container[i])
    end
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::AbstractMatrix{T}) where {T <: Real}
    @views for (i, col) in enumerate(eachcol(dist.a))
        rand!(rng, Dirichlet(col), container[:, i])
    end
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::AbstractVector{T}) where {T <: AbstractMatrix}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function BayesBase.logpdf(dist::MatrixDirichlet, x::Matrix)
    return vmapreduce(
        d -> logpdf(Dirichlet(convert(Vector, d[1])), convert(Vector, d[2])),
        +,
        zip(eachcol(dist.a), eachcol(x))
    )
end

BayesBase.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

BayesBase.mean(::Base.Broadcast.BroadcastFunction{typeof(log)}, dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

BayesBase.default_prod_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a - Ones{T}(size(left.a)))
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{MatrixDirichlet}, x)
    l = length(getnaturalparameters(ef))
    ## The columns of x should be normalized. all(≈(1), sum(eachrow(x))) is a convenient way of doing that
    ## because eachrow(x) will return row slices and sum will take the sum of the row slices along the first dimension
    return l == length(x) && !any(x -> x < zero(x), x) && all(≈(1), sum(eachrow(x)))
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{MatrixDirichlet}, η, conditioner) =
    isnothing(conditioner) && length(η) > 1 && all(isless.(-1, η)) && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{MatrixDirichlet}, θ, conditioner) =
    isnothing(conditioner) && length(θ) > 1 && all(>(0), θ) && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{MatrixDirichlet})(tuple_of_θ::Tuple{Any})
    (α,) = tuple_of_θ
    return (α - Ones{Float64}(size(α)),)
end

function (::NaturalToMean{MatrixDirichlet})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (η + Ones{Float64}(size(η)),)
end

function unpack_parameters(::Type{MatrixDirichlet}, packed)
    n = length(packed)
    return (reshape(view(packed, 1:n), isqrt(n), isqrt(n)),)
end

isbasemeasureconstant(::Type{MatrixDirichlet}) = ConstantBaseMeasure()

getbasemeasure(::Type{MatrixDirichlet}) = (x) -> one(Float64)
getsufficientstatistics(::Type{MatrixDirichlet}) = (x -> vmap(log, x),)

getlogpartition(::NaturalParametersSpace, ::Type{MatrixDirichlet}) =
    (η) -> begin
        (η1,) = unpack_parameters(MatrixDirichlet, η)
        return vmapreduce(
            d -> getlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)),
            +,
            eachcol(η1)
        )
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{MatrixDirichlet}) =
    (η) -> begin
        (η1,) = unpack_parameters(MatrixDirichlet, η)
        return vmapreduce(
            d -> getgradlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)), vcat,
            eachcol(η1))
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MatrixDirichlet}) =
    (η) -> begin
        (η1,) = unpack_parameters(MatrixDirichlet, η)
        ones = Ones{Float64}(size(η1))
        ηp1 = η1 + ones

        matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * ones),
            Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

        return vmapreduce(identity, blockdiag, matrices)
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{MatrixDirichlet}) =
    (θ) -> begin
        (α,) = unpack_parameters(MatrixDirichlet, θ)
        return vmapreduce(
            d -> getlogpartition(MeanParametersSpace(), Dirichlet)(convert(Vector, d)),
            +,
            eachcol(α)
        )
    end

getgradlogpartition(::MeanParametersSpace, ::Type{MatrixDirichlet}) =
    (θ) -> begin
        (α,) = unpack_parameters(MatrixDirichlet, θ)
        return vmapreduce(
            d -> getgradlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)), vcat,
            eachcol(α))
    end

getfisherinformation(::MeanParametersSpace, ::Type{MatrixDirichlet}) =
    (θ) -> begin
        (α,) = unpack_parameters(MatrixDirichlet, θ)
        matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * Ones{Float64}(size(α))),
            Iterators.zip(map(d -> trigamma(d), sum(α, dims = 1)), map(d -> trigamma.(d), eachcol(α))))

        return vmapreduce(identity, blockdiag, matrices)
    end
