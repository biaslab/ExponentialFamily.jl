module LaplaceTest

using Test
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Zygote
using ForwardDiff

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, ExponentialFamilyDistribution, logpartition,
    basemeasure, getbasemeasure, getnaturalparameters, getsufficientstatistics, getconditioner, fisherinformation,
    logpdf, support, getsupport

@testset "Laplace" begin
    @testset "vague" begin
        d = vague(Laplace)

        @test typeof(d) <: Laplace
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1e12)
    end

    @testset "prod" begin
        for i in 1:100
            μleft = 100 * randn()
            μright = 100 * randn()
            σleft = 100 * rand()
            σright = 100 * rand()
            l_left = Laplace(μleft, σleft)
            l_right = Laplace(μleft, σright)
            l_right2 = Laplace(μright, σright)
            ef_left = convert(ExponentialFamilyDistribution, l_left)
            ef_right = convert(ExponentialFamilyDistribution, l_right)
            ef_right2 = convert(ExponentialFamilyDistribution, l_right2)
            (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
            (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
            (η_right2, conditioner_right2) = (getnaturalparameters(ef_right2), getconditioner(ef_right2))
            @test prod(ef_left, ef_right) ==
                  ExponentialFamilyDistribution(Laplace, η_left + η_right, conditioner_left)
            @test prod(ClosedProd(), l_left, l_right) ≈ convert(Distribution, prod(ef_left, ef_right))

            basemeasure = (x) -> 1.0
            sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right2)]
            sorted_conditioner = sort([conditioner_left, conditioner_right2])
            function logpartition(η)
                A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right2)
                A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right2)
                A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right2)
                B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
                B2 =
                    (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
                    (η[1] - η[2])
                B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

                return log(A1 * B1 + A2 * B2 + A3 * B3)
            end
            naturalparameters = vcat(η_left, η_right2)
            supp = support(l_left)
            dist_prod = prod(ClosedProd(), l_left, l_right2)
            ef_prod = prod(ef_left, ef_right2)
            @test getnaturalparameters(dist_prod) == naturalparameters
            @test getsupport(dist_prod) == supp
            @test getbasemeasure(dist_prod)(1.0) == basemeasure(1.0)
            @test getsufficientstatistics(dist_prod)(1.0) ==
                  sufficientstatistics(1.0)

            @test getnaturalparameters(ef_prod) == naturalparameters
            @test getsupport(ef_prod) == supp
            @test getbasemeasure(ef_prod)(1.0) == basemeasure(1.0)
            @test getsufficientstatistics(ef_prod)(1.0) ==
                  sufficientstatistics(1.0)
        end
    end

    @testset "natural parameters related" begin
        @testset "convert" begin
            for i in 1:10
                @test convert(Distribution, ExponentialFamilyDistribution(Laplace, [-i], 2.0)) ==
                      Laplace(2.0, 1 / i)

                @test convert(ExponentialFamilyDistribution, Laplace(sqrt(i), i)) ==
                      ExponentialFamilyDistribution(Laplace, [-1 / i], sqrt(i))
            end
        end

        @testset "logpartition" begin
            @test logpartition(ExponentialFamilyDistribution(Laplace, [-1.0], 1.0)) ≈ log(2)
            @test logpartition(ExponentialFamilyDistribution(Laplace, [-2.0], 1.0)) ≈ log(1)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(ExponentialFamilyDistribution(Laplace, [-i], 0.0), 0.01) ≈
                      logpdf(Laplace(0.0, 1 / i), 0.01)
                @test logpdf(ExponentialFamilyDistribution(Laplace, [-i], 1.0), 0.5) ≈
                      logpdf(Laplace(1.0, 1 / i), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(Laplace, [-i], 1.0)) === true
                @test isproper(ExponentialFamilyDistribution(Laplace, [i], 2.0)) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(ExponentialFamilyDistribution(Laplace, [-i], 1.0), i^2) == 1.0
            end
        end

        @testset "fisher information" begin
            for λ in 1:10, u in 1.0:0.5:5.0
                dist = Laplace(u, λ)
                ef = convert(ExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)
                transformation(η) = [u, -inv(η[1])]
                f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Laplace, η, getconditioner(ef)))
                autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
                @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-8
                J = ForwardDiff.jacobian(transformation, η)
                @test first(J' * fisherinformation(dist) * J) ≈ first(fisherinformation(ef)) atol = 1e-8
            end
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for λ in 1:10, u in 1.0:0.5:5.0
            dist = Laplace(u, λ)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
