
# Poisson comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Poisson: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    @testset for i in 2:7
        @testset let d = Poisson(2 * (i + 1))
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
            η1 = first(getnaturalparameters(ef))

            for x in 1:5
                @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1 / factorial(x)
                @test @inferred(logbasemeasure(ef, x)) === -loggamma(x + one(x))
                @test @inferred(sufficientstatistics(ef, x)) === (x,)
                @test @inferred(logpartition(ef)) ≈ exp(η1)
            end
        end
    end

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, Poisson, [Inf])
        @test !isproper(space, Poisson, [NaN])
        @test !isproper(space, Poisson, [1.0], NaN)
        @test !isproper(space, Poisson, [0.5, 0.5], 1.0)
    end
    ## mean parameter should be integer in the MeanParametersSpace
    @test !isproper(MeanParametersSpace(), Poisson, [-0.1])
    @test_throws Exception convert(ExponentialFamilyDistribution, Poisson(Inf))
end

@testitem "Poisson: prod" begin
    include("distributions_setuptests.jl")

    @testset for λleft in 2:3, λright in 3:4
        left = Poisson(λleft)
        right = Poisson(λright)
        prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
        sample_points = collect(1:5)
        for x in sample_points
            @test basemeasure(prod_dist, x) == (1 / gamma(x + one(x))^2)
            @test sufficientstatistics(prod_dist, x) == (x,)
        end
        sample_points = [-5, -2, 0, 2, 5]
        for η in sample_points
            @test logpartition(prod_dist, η) == log(abs(besseli(0, 2 * exp(η / 2))))
        end
        @test getnaturalparameters(prod_dist) == [log(λleft) + log(λright)]
        @test getsupport(prod_dist) == NaturalNumbers()

        @test sum(pdf(prod_dist, x) for x in 0:15) ≈ 1.0
        @test value_support(prod_dist) == Discrete
    end
end
