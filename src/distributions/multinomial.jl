export Multinomial

import Distributions: Multinomial, probs 

vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

probvec(dist::Multinomial) = probs(dist) #return probability vector

function convert_eltype(::Type{Multinomial}, ::Type{T}, distribution::Multinomial{R}) where {T <: Real, R <: Real}
    n,p = params(distribution)
    return Multinomial(n, convert(AbstractVector{T}, p))
end

prod_analytical_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Multinomial, right::Multinomial)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)

    return Multinomial(left.n, mvec ./ norm)
end


# Standard parameters to natural parameters
function Base.convert(::Type{NaturalParameters}, dist::Multinomial)
    n, p = params(dist)
    logprobabilities = log.(p)
    # The params of NaturalParameters(Multinomial) is a Tuple (n, log(probvec))
    return NaturalParameters(Multinomial, (n,logprobabilities))
end

function Base.convert(::Type{Distribution}, η::NaturalParameters{Multinomial})
    return Multinomial(first(get_params(η)), softmax(last(get_params(η))))
end

check_valid_natural(::Type{<:Multinomial}, params) = length(last(params)) >= 2

function isproper(params::NaturalParameters{Multinomial})
    η = get_params(params)
    n = first(η)
    p = last(η)
    return (n >= 1) && (length(p)>=2) && (sum(p) == 1)
end

lognormalizer(::NaturalParameters{Multinomial}) = 0.0

function basemeasure(::Union{<:NaturalParameters{Multinomial}, <:Multinomial}, x) 
    """
    x is a vector satisfying ∑x = n
    """
    n = Int(sum(x))
    return factorial(n) / prod(factorial.(x))
end

function Base.:+(left::NaturalParameters{Multinomial}, right::NaturalParameters{Multinomial}) 
    η_left = get_params(left)
    η_right = get_params(right)
    
    return NaturalParameters(Multinomial, (first(η_left),last(η_left) + last(η_right)))
end
    
function Base.:-(left::NaturalParameters{Multinomial}, right::NaturalParameters{Multinomial}) 
    η_left = get_params(left)
    η_right = get_params(right)
    
    return NaturalParameters(Multinomial, (first(η_left),last(η_left) - last(η_right)))
end