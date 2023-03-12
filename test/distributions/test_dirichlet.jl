module DirichletTest

using Test
using ExponentialFamily
using Distributions
using Random
import ExponentialFamily: NaturalParameters, get_params
import SpecialFunctions: loggamma
@testset "Dirichlet" begin

    # Dirichlet comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Dirichlet)

        d1 = vague(Dirichlet, 2)

        @test typeof(d1) <: Dirichlet
        @test probvec(d1) == ones(2)

        d2 = vague(Dirichlet, 4)

        @test typeof(d2) <: Dirichlet
        @test probvec(d2) == ones(4)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Dirichlet([1.0, 1.0, 1.0]), Dirichlet([1.0, 1.0, 1.0])) ==
              Dirichlet([1.0, 1.0, 1.0])
        @test prod(ProdAnalytical(), Dirichlet([1.1, 1.0, 2.0]), Dirichlet([1.0, 1.2, 1.0])) ==
              Dirichlet([1.1, 1.2000000000000002, 2.0])
        @test prod(ProdAnalytical(), Dirichlet([1.1, 2.0, 2.0]), Dirichlet([3.0, 1.2, 5.0])) ==
              Dirichlet([3.0999999999999996, 2.2, 6.0])
    end

    @testset "probvec" begin
        @test probvec(Dirichlet([1.0, 1.0, 1.0])) == [1.0, 1.0, 1.0]
        @test probvec(Dirichlet([1.1, 2.0, 2.0])) == [1.1, 2.0, 2.0]
        @test probvec(Dirichlet([3.0, 1.2, 5.0])) == [3.0, 1.2, 5.0]
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Dirichlet([1.0, 1.0, 1.0])) ≈ [-1.5000000000000002, -1.5000000000000002, -1.5000000000000002]
        @test mean(log, Dirichlet([1.1, 2.0, 2.0])) ≈ [-1.9517644694670657, -1.1052251939575213, -1.1052251939575213]
        @test mean(log, Dirichlet([3.0, 1.2, 5.0])) ≈ [-1.2410879175727905, -2.4529121492634465, -0.657754584239457]
    end

    @testset "promote_variate_type" begin
        @test_throws MethodError promote_variate_type(Univariate, Dirichlet)

        @test promote_variate_type(Multivariate, Dirichlet) === Dirichlet
        @test promote_variate_type(Matrixvariate, Dirichlet) === MatrixDirichlet

        @test promote_variate_type(Multivariate, MatrixDirichlet) === Dirichlet
        @test promote_variate_type(Matrixvariate, MatrixDirichlet) === MatrixDirichlet
    end

    @testset "NaturalParameters" begin
        @test convert(NaturalParameters, Dirichlet([0.6, 0.7 ])) == NaturalParameters(Dirichlet,[0.6, 0.7 ] .- 1)
        b_01 = Dirichlet([10.0, 10.0, 10.0])
        @test lognormalizer(convert(NaturalParameters, Dirichlet([1, 1]))) ≈ 2loggamma(2)
        @test lognormalizer(convert(NaturalParameters, Dirichlet([0.1, 0.2]))) ≈ loggamma(0.1)+loggamma(0.2) - loggamma(0.3)
        for i in 1:9
            b = Dirichlet([i / 10.0, i/5, i])
            bnp = convert(NaturalParameters, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, [0.5, 0.4, 0.1]) ≈ logpdf(b, [0.5, 0.4, 0.1])
            @test logpdf(bnp, [0.2, 0.3, 0.5]) ≈ logpdf(b, [0.2, 0.3, 0.5])

            @test convert(NaturalParameters, b) == bnp

            @test prod(ProdAnalytical(), convert(Distribution, convert(NaturalParameters, b_01) - bnp), b) ≈ b_01
        end
        @test isproper(NaturalParameters(Dirichlet, [10,2,3])) === true
        @test isproper(NaturalParameters(Dirichlet, [-0.1,-0.2,3])) === true
        @test isproper(NaturalParameters(Dirichlet, [-0.1,-0.2,-3])) === false
    end
end

end
