export InverseWishart

import Distributions: InverseWishart, Wishart, pdf!
import Base: ndims, size, convert
import LinearAlgebra
import StatsFuns: logπ, logmvgamma
import SpecialFunctions: digamma, loggamma

"""
InverseWishartFast{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution

The `InverseWishartFast` struct represents an improper Inverse Wishart distribution. It is similar to the `InverseWishart` distribution from `Distributions.jl`, but it does not check input arguments, allowing the creation of improper `InverseWishart` messages. 

For model creation and regular usage, it is recommended to use `InverseWishart` from `Distributions.jl`. The `InverseWishartFast` distribution is intended for internal purposes and should not be directly used by regular users.

# Fields
- `ν::T`: The degrees of freedom parameter of the inverse Wishart distribution.
- `S::A`: The scale matrix parameter of the inverse Wishart distribution.

## Note

The `InverseWishartFast` distribution does not enforce input argument validation, making it suitable for specialized cases where improper message constructions are needed.
"""
struct InverseWishartFast{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function InverseWishartFast(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return InverseWishartFast(convert(T, ν), convert(AbstractArray{T}, S))
end

InverseWishartFast(ν::Integer, S::AbstractMatrix{Real}) = InverseWishartFast(float(ν), S)

Distributions.params(dist::InverseWishartFast) = (dist.ν, dist.S)
Distributions.mean(dist::InverseWishartFast)   = mean(convert(InverseWishart, dist))
Distributions.var(dist::InverseWishartFast)    = var(convert(InverseWishart, dist))
Distributions.cov(dist::InverseWishartFast)    = cov(convert(InverseWishart, dist))
Distributions.mode(dist::InverseWishartFast)   = mode(convert(InverseWishart, dist))

mean_cov(dist::InverseWishartFast) = mean_cov(convert(InverseWishart, dist))

Base.size(dist::InverseWishartFast)           = size(dist.S)
Base.size(dist::InverseWishartFast, dim::Int) = size(dist.S, dim)

const InverseWishartDistributionsFamily{T} = Union{InverseWishart{T}, InverseWishartFast{T}}

to_marginal(dist::InverseWishartFast) = convert(InverseWishart, dist)

function Base.convert(::Type{InverseWishartFast{T}}, distribution::InverseWishartFast) where {T}
    (ν, S) = params(distribution)
    return InverseWishartFast(convert(T, ν), convert(AbstractMatrix{T}, S))
end

# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InverseWishartFast)
    d = size(dist, 1)
    ν, S = params(dist)
    d * (d - 1) / 4 * logπ + mapreduce(i -> loggamma((ν + 1.0 - i) / 2), +, 1:d) + ν / 2 * d +
    (d + 1) / 2 * (logdet(S) - log(2)) -
    (ν + d + 1) / 2 * mapreduce(i -> digamma((ν - d + i) / 2), +, 1:d)
end

function Distributions.mean(::typeof(logdet), dist::InverseWishartFast)
    d = size(dist, 1)
    ν, S = params(dist)
    return -(mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(S))
end

function Distributions.mean(::typeof(inv), dist::InverseWishartFast)
    return mean(cholinv, dist)
end

function Distributions.mean(::typeof(cholinv), dist::InverseWishartFast)
    ν, S = params(dist)
    return mean(Wishart(ν, cholinv(S)))
end

function Distributions.rand(rng::AbstractRNG, sampleable::InverseWishartFast{T}, n::Int) where {T}
    container = [Matrix{T}(undef, size(sampleable)) for _ in 1:n]
    return rand!(rng, sampleable, container)
end

function Distributions.rand!(rng::AbstractRNG, sampleable::InverseWishartFast, x::AbstractVector{<:AbstractMatrix})
    # This is an adapted version of sampling from Distributions.jl
    (df, S⁻¹) = Distributions.params(sampleable)
    S = cholinv(S⁻¹)
    L = Distributions.PDMats.chol_lower(fastcholesky(S))

    p = size(S, 1)
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(ArgumentError("df of a singular Wishart distribution must be an integer (got $df)"))
    end

    A     = similar(S)
    l     = length(S)
    axes2 = axes(A, 2)
    r     = rank(S)

    for C in x
        if singular
            randn!(rng, view(A, :, view(axes2, 1:r)))
            fill!(view(A, :, view(axes2, (r+1):lastindex(axes2))), zero(eltype(A)))
        else
            Distributions._wishart_genA!(rng, A, df)
        end
        # Distributions.unwhiten!(S, A)
        lmul!(L, A)

        M = Cholesky(A, 'L', convert(LinearAlgebra.BlasInt, 0))
        LinearAlgebra.inv!(M)

        copyto!(C, 1, M.L.data, 1, l)
    end

    return x
end

function Distributions.pdf!(
    out::AbstractArray{<:Real},
    distribution::InverseWishartFast,
    samples::AbstractArray{<:AbstractMatrix{<:Real}, O}
) where {O}
    @assert length(out) === length(samples) "Invalid dimensions in pdf!"

    p = size(distribution, 1)
    (df, Ψ) = Distributions.params(distribution)

    T = copy(Ψ)
    R = similar(T)
    l = length(T)

    M = fastcholesky!(T)

    h_df = df / 2
    Ψld = logdet(M)
    logc0 = -h_df * (p * convert(typeof(df), Distributions.logtwo) - Ψld) - logmvgamma(p, h_df)

    @inbounds for i in 1:length(out)
        copyto!(T, 1, samples[i], 1, l)
        C = fastcholesky!(T)
        ld = logdet(C)
        LinearAlgebra.inv!(C)
        mul!(R, Ψ, C.factors)
        r = tr(R)
        out[i] = exp(-0.5 * ((df + p + 1) * ld + r) + logc0)
    end

    return out
end

vague(::Type{<:InverseWishart}, dims::Integer) = InverseWishart(dims + 2, tiny .* diageye(dims))

Base.ndims(dist::InverseWishart) = size(dist, 1)

function Base.convert(::Type{InverseWishart}, dist::InverseWishartFast)
    (ν, S) = params(dist)
    return InverseWishart(ν, Matrix(Hermitian(S)))
end

Base.convert(::Type{InverseWishartFast}, dist::InverseWishart) = InverseWishartFast(params(dist)...)

function logpdf_sample_optimized(dist::InverseWishart)
    optimized_dist = convert(InverseWishartFast, dist)
    return (optimized_dist, optimized_dist)
end

# We do not define prod between `InverseWishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `InverseWishartFast` messages as they are significantly faster in creation
closed_prod_rule(::Type{<:InverseWishartFast}, ::Type{<:InverseWishartFast}) = ClosedProd()

function Base.prod(::ClosedProd, left::InverseWishartFast, right::InverseWishartFast)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two InverseWishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V = lS + rS

    df = ldf + rdf + d + 1

    return InverseWishartFast(df, V)
end

function pack_naturalparameters(dist::Union{InverseWishartFast, InverseWishart})
    dof, scale = params(dist)
    p = first(size(scale))

    return vcat(-(dof + p + 1) / 2, vec(-scale / 2))
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:InverseWishartFast})
    η = getnaturalparameters(ef)
    len = length(η)
    n = Int64(isqrt(len - 1))
    @inbounds η1 = η[1]
    @inbounds η2 = reshape(view(η, 2:len), n, n)

    return η1, η2
end

check_valid_natural(::Type{<:Union{InverseWishartFast, InverseWishart}}, params) = length(params) >= 5

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Union{InverseWishartFast, InverseWishart}) =
    ExponentialFamilyDistribution(InverseWishartFast, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{<:InverseWishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    return InverseWishart(-(2 * η1 + p + 1), -2 * η2)
end

function logpartition(ef::ExponentialFamilyDistribution{<:InverseWishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    term1 = (η1 + (p + 1) / 2) * logdet(-η2)
    term2 = logmvgamma(p, -(η1 + (p + 1) / 2))
    return term1 + term2
end

function isproper(ef::ExponentialFamilyDistribution{<:InverseWishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    isposdef(-η2) && (η1 < 0)
end

function fisherinformation(ef::ExponentialFamilyDistribution{<:InverseWishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    invη2 = inv(η2)
    return [mvtrigamma(p, (η1 + (p + 1) / 2)) -vec(invη2)'; -vec(invη2) (η1+(p+1)/2)*kron(invη2, invη2)]
end

function fisherinformation(dist::InverseWishart)
    ν = dist.df
    S = dist.Ψ
    p = first(size(S))
    invscale = inv(S)

    hessian = ones(eltype(S), p^2 + 1, p^2 + 1)
    hessian[1, 1] = mvtrigamma(p, -ν / 2) / 4
    hessian[1, 2:p^2+1] = as_vec(invscale) / 2
    hessian[2:p^2+1, 1] = as_vec(invscale) / 2
    hessian[2:p^2+1, 2:p^2+1] = ν / 2 * kron(invscale, invscale)
    hessian[2:p^2+1, 2:p^2+1] = -1 * hessian[2:p^2+1, 2:p^2+1]
    return hessian
end

function insupport(ef::ExponentialFamilyDistribution{InverseWishartFast, P, C, Safe}, x::Matrix) where {P, C}
    return size(getindex(unpack_naturalparameters(ef), 2)) == size(x) && isposdef(x)
end

basemeasure(::ExponentialFamilyDistribution{<:InverseWishartFast}) = one(Float64)
function basemeasure(
    ::ExponentialFamilyDistribution{<:InverseWishartFast},
    x
)
    return one(eltype(x))
end

sufficientstatistics(ef::ExponentialFamilyDistribution{<:InverseWishartFast}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(
    ::ExponentialFamilyDistribution{<:InverseWishartFast},
    x
)
    return vcat(chollogdet(x), vec(cholinv(x)))
end
