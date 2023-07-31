module MvNormalWishartTests

using Distributions, HCubature
using Test
using StatsFuns: logmvgamma
using LinearAlgebra
using ForwardDiff
using ExponentialFamily
import ExponentialFamily:
    MvNormalWishart, ExponentialFamilyDistribution, params, dof, invscatter, reconstructargument!
import ExponentialFamily:
    scale, dim, getnaturalparameters, tiny, logpartition, cholinv, MvNormalMeanPrecision, sufficientstatistics, as_vec
using Distributions
using Random

function normal_wishart_pdf(x::Vector{Float64},
    lambda::Matrix{Float64},
    mu::Vector{Float64},
    kappa::Float64,
    nu::Float64,
    Ψ::Matrix{Float64})
    return pdf(MvNormalMeanPrecision(mu, kappa * lambda), x) * pdf(Wishart(nu, Ψ), lambda)
end

# Derrivative-friendly function for the natural parameters of the MvNormalWishart distribution
logpartition(::ExponentialFamilyDistribution{T}, ηvec::Vector{F}) where 
        {T, F <: Real} = logpartition(ExponentialFamilyDistribution(T, ηvec))


@testset "MvNormalWishart" begin
    @testset "common" begin
        m = rand(2)
        dist = MvNormalWishart(m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
        @test params(dist) == (m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
        @test dof(dist) == 3.0
        @test invscatter(dist) == [1.0 0.0; 0.0 1.0]
        @test scale(dist) == 0.1
        @test dim(dist) == 2
    end

    @testset "conversions" begin
        for i in 1:10, j in 2:6
            m = rand(j)
            κ = rand()
            Ψ = diagm(rand(j))
            ν = 2 * j + 1
            dist = MvNormalWishart(m, Ψ, κ, ν)
            ef = convert(ExponentialFamilyDistribution, dist)

            @test getnaturalparameters(ef) ≈ vcat(κ * m, vec(-(1 / 2) * (inv(Ψ) + κ * m*m')), -κ / 2, (ν - j) / 2)
            @test invscatter(convert(Distribution, ef)) ≈ cholinv(Ψ)
            @test dof(convert(Distribution, ef)) == 2 * j + 1
        end
    end

    @testset "exponential family functions" begin
        for i in 1:10, j in 2:5, κ in 0.01:1.0:5.0
            m = rand(j)
            Ψ = m * m' + I
            dist = MvNormalWishart(m, Ψ, κ, j + 1)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test pdf(dist, [m, Ψ]) ≈ normal_wishart_pdf(m, Ψ, m, κ, float(j + 1), Ψ)
            @test logpdf(dist, [m, Ψ]) ≈ log(normal_wishart_pdf(m, Ψ, m, κ, float(j + 1), Ψ))
        end
    end

    @testset "sampling" begin
        nsamples = 100
        for j in 2:5, κ in 1:5
            m = rand(j)
            Ψ = m * m' + I
            dist = MvNormalWishart(m, Ψ, κ, j + 3)
            ef = convert(ExponentialFamilyDistribution, dist)
            st = sufficientstatistics(dist)
            samples = rand(MersenneTwister(j), dist, nsamples)
            ηvec = getnaturalparameters(ef)
            expsuffstats = sum(st(sample[1], sample[2]) for sample in samples) / nsamples
            expsuffstatsvec = ForwardDiff.gradient(x -> logpartition(ef, x), ηvec)
            @test expsuffstats ≈  expsuffstatsvec rtol = 0.1
        end
    end

    @testset "prod" begin
        for j in 2:2, κ in 1:2
            m1 = rand(j)
            m2 = rand(j)
            Ψ1 = m1 * m1' + I
            Ψ2 = m2 * m2' + I
            dist1 = MvNormalWishart(m1, Ψ1, κ, j + 3)
            dist2 = MvNormalWishart(m2, Ψ2, κ, j + 3)
            ef1 = convert(ExponentialFamilyDistribution, dist1)
            ef2 = convert(ExponentialFamilyDistribution, dist2)
            @test prod(ClosedProd(), dist1, dist2) == convert(Distribution, prod(ClosedProd(), ef1, ef2))
        end
    end
end

end
