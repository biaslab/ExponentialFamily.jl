module WishartTest

using Test
using ExponentialFamily
using Distributions
using Random
using LinearAlgebra
using StableRNGs
using ForwardDiff

import ExponentialFamily: WishartImproper, KnownExponentialFamilyDistribution, reconstructargument!,
    getnaturalparameters, basemeasure, fisherinformation, logpartition, as_vec
import StatsFuns: logmvgamma

function logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec::Vector{F}) where {T, F <: Real}
    ηef = getnaturalparameters(ef)
    reconstructargument!(ηef, ηef, ηvec)
    return logpartition(KnownExponentialFamilyDistribution(T, ηef))
end

function transformation(params)
    η1, η2 = params[1], params[2:end]
    p = Int(sqrt(length(η2)))
    η2 = reshape(η2, (p, p))
    return [2 * η1 + p + 1; as_vec(0.5inv(-η2))]
end

@testset "Wishart" begin

    # Wishart comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "mean(::logdet)" begin
        @test mean(logdet, Wishart(3, [1.0 0.0; 0.0 1.0])) ≈ 0.845568670196936
        @test mean(
            logdet,
            Wishart(
                5,
                [
                    1.4659658963311604 1.111775094889733 0.8741034114800605
                    1.111775094889733 0.8746971141492232 0.6545661366809246
                    0.8741034114800605 0.6545661366809246 0.5498917856395482
                ]
            )
        ) ≈ -3.4633310802040693
    end

    @testset "mean(::cholinv)" begin
        L    = rand(2, 2)
        S    = L * L' + diageye(2)
        invS = cholinv(S)
        @test mean(inv, Wishart(5, S)) ≈ mean(InverseWishart(5, invS))
        @test mean(cholinv, Wishart(5, S)) ≈ mean(InverseWishart(5, invS))
    end

    @testset "vague" begin
        @test_throws MethodError vague(Wishart)

        d = vague(Wishart, 3)

        @test typeof(d) <: Wishart
        @test mean(d) == Matrix(Diagonal(3 * 1e12 * ones(3)))
    end

    @testset "rand" begin
        for d in (2, 3, 4, 5)
            v = rand() + d
            L = rand(d, d)
            S = L' * L + d * diageye(d)
            invS = cholinv(S)
            cS = copy(S)
            cinvS = copy(invS)
            container1 = [zeros(d, d) for _ in 1:100]
            container2 = [zeros(d, d) for _ in 1:100]

            # Check inplace versions
            @test rand!(StableRNG(321), Wishart(v, S), container1) ≈
                  rand!(StableRNG(321), WishartImproper(v, invS), container2)

            # Check that matrices are not corrupted
            @test all(S .=== cS)
            @test all(invS .=== cinvS)

            # Check non-inplace versions
            @test rand(StableRNG(321), Wishart(v, S), length(container1)) ≈
                  rand(StableRNG(321), WishartImproper(v, invS), length(container2))
        end
    end

    @testset "prod" begin
        inv_v1 = cholinv([9.0 -3.4; -3.4 11.0])
        inv_v2 = cholinv([10.2 -3.3; -3.3 5.0])
        inv_v3 = cholinv([8.1 -2.7; -2.7 9.0])

        @test prod(ClosedProd(), WishartImproper(3, inv_v1), WishartImproper(3, inv_v2)) ≈
              WishartImproper(
            3,
            cholinv([4.776325721474591 -1.6199382410125422; -1.6199382410125422 3.3487476649765537])
        )
        @test prod(ClosedProd(), WishartImproper(4, inv_v1), WishartImproper(4, inv_v3)) ≈
              WishartImproper(
            5,
            cholinv([4.261143738311623 -1.5064864332819319; -1.5064864332819319 4.949867121624725])
        )
        @test prod(ClosedProd(), WishartImproper(5, inv_v2), WishartImproper(4, inv_v3)) ≈
              WishartImproper(6, cholinv([4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499]))
    end

    @testset "ndims" begin
        @test ndims(vague(Wishart, 3)) === 3
        @test ndims(vague(Wishart, 4)) === 4
        @test ndims(vague(Wishart, 5)) === 5
    end

    @testset "natural parameters related" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(
                    Distribution,
                    KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]])
                ) ≈
                      Wishart(9.0, -0.5 * inv([-i 0.0; 0.0 -i]))
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                wishart_np = KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]])
                distribution = Wishart(9.0, -0.5 * inv([-i 0.0; 0.0 -i]))
                @test logpdf(distribution, [1.0 0.0; 0.0 1.0]) ≈ logpdf(wishart_np, [1.0 0.0; 0.0 1.0])
                @test logpdf(distribution, [1.0 0.2; 0.2 1.0]) ≈ logpdf(wishart_np, [1.0 0.2; 0.2 1.0])
                @test logpdf(distribution, [1.0 -0.1; -0.1 3.0]) ≈ logpdf(wishart_np, [1.0 -0.1; -0.1 3.0])
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-1.0 0.0; 0.0 -1.0]])) ≈
                  logmvgamma(2, 3.0 + (2 + 1) / 2)
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]])) === true
                @test isproper(KnownExponentialFamilyDistribution(WishartImproper, [3.0, [i 0.0; 0.0 -i]])) === false
                @test isproper(KnownExponentialFamilyDistribution(WishartImproper, [-1.0, [-i 0.0; 0.0 -i]])) === false
            end
        end

        @testset "basemeasure" begin
            for i in 1:10
                @test_throws AssertionError basemeasure(
                    KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]]),
                    rand(3, 3)
                )
                L = rand(2, 2)
                @test basemeasure(
                    KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]]),
                    L * L'
                ) == 1.0
            end
        end

        @testset "base operations" begin
            for i in 1:10
                np1 = KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-i 0.0; 0.0 -i]])
                np2 = KnownExponentialFamilyDistribution(WishartImproper, [3.0, [-2i 0.0; 0.0 -2i]])
                @test prod(np1, np2) == KnownExponentialFamilyDistribution(
                    WishartImproper,
                    [3.0, [-2i 0.0; 0.0 -2i]] + [3.0, [-i 0.0; 0.0 -i]]
                )
            end
        end

        @testset "fisher information" begin
            rng = StableRNG(42)
            for df in 2:20
                L = randn(rng, df, df)
                A = L * L' + 1e-8 * diageye(df)
                dist = Wishart(df, A)
                ef = convert(KnownExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)
                η_vec = vcat(η[1], as_vec(η[2]))
                fef = fisherinformation(ef)
                fdist = fisherinformation(dist)
                ## We do not test the analytic solution agains autograd because autograd hessian return values that are permuted and
                ## causes fisherinformation to be non-positive definite.
                J = ForwardDiff.jacobian(transformation, η_vec)
                @test fef ≈ J' * fdist * J rtol = 1e-8

                f_logpartition = (η_vec) -> logpartition(ef, η_vec)
                autograd_information = (η_vec) -> ForwardDiff.hessian(f_logpartition, η_vec)
                ag_fi = autograd_information(η_vec)

                svdfisheref = svd(fef)
                svdautograd = svd(ag_fi)
                ## Julia returns very small complex values which causes problems. Therefore we take the real parts. 
                eigenfisheref = real.(eigvals(fef))
                @test all(x -> x > 0, eigenfisheref)
                @test svdfisheref.S ≈ svdautograd.S
            end
        end
    end
    @testset "KnownExponentialFamilyDistribution mean,cov" begin
        rng = StableRNG(42)
        for df in 2:20
            L = randn(rng, df, df)
            A = L * L' + 1e-8 * diageye(df)
            dist = Wishart(df, A)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) rtol = 1e-8
            @test cov(dist) ≈ cov(ef) rtol = 1e-8
        end
    end
end

end
