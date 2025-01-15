
# Binomial comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Binomial: probvec" begin
    include("distributions_setuptests.jl")

    @test all(probvec(Binomial(2, 0.8)) .≈ (0.2, 0.8))
    @test probvec(Binomial(2, 0.2)) == (0.8, 0.2)
    @test probvec(Binomial(2, 0.1)) == (0.9, 0.1)
    @test probvec(Binomial(2)) == (0.5, 0.5)
end

@testitem "Binomial: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Binomial)
    @test_throws MethodError vague(Binomial, 1 / 2)

    vague_dist = vague(Binomial, 5)
    @test typeof(vague_dist) <: Binomial
    @test probvec(vague_dist) == (0.5, 0.5)
end

@testitem "Binomial: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for n in (2, 3, 4), p in 0.1:0.2:0.9
        @testset let d = Binomial(n, p)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

            η₁ = log(p / (1 - p))

            for x in 0:n
                @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === binomial(n, x)
                @test @inferred(logbasemeasure(ef, x)) === loggamma(n + 1) - (loggamma(n - x + 1) + loggamma(x + 1))
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x,))
                @test @inferred(logpartition(ef)) ≈ (n * log(1 + exp(η₁)))
            end

            @test !@inferred(insupport(ef, -1))
            @test @inferred(insupport(ef, 0))

            # Not in the support
            @test_throws Exception logpdf(ef, -1)
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), Binomial, [-1], 1)
    @test !isproper(MeanParametersSpace(), Binomial, [0.5], -1)
    @test !isproper(MeanParametersSpace(), Binomial, [-0.1, 1], 10)
    @test !isproper(NaturalParametersSpace(), Binomial, [-1.1], -1)
    @test !isproper(NaturalParametersSpace(), Binomial, [1, -1.1], 10)
end

@testitem "Binomial: prod ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for nleft in 1:1, pleft in 0.1:0.1:0.1, nright in 1:1, pright in 0.1:0.1:0.1
        @testset let (left, right) = (Binomial(nleft, pleft), Binomial(nright, pright))
            for (efleft, efright) in ((left, right), (convert(ExponentialFamilyDistribution, left), convert(ExponentialFamilyDistribution, right)))
                for strategy in (PreserveTypeProd(ExponentialFamilyDistribution),)
                    prod_dist = prod(strategy, efleft, efright)

                    @test prod_dist isa ExponentialFamilyDistribution

                    hist_sum(x) =
                        basemeasure(prod_dist, x) * exp(
                            dot(ExponentialFamily.flatten_parameters(sufficientstatistics(prod_dist, x)), getnaturalparameters(prod_dist)) -
                            logpartition(prod_dist)
                        )

                    support = 0:1:max(nleft, nright)

                    @test sum(hist_sum, support) ≈ 1.0 atol = 1e-9
                    @test value_support(prod_dist) == Discrete
                    for x in support
                        @test basemeasure(prod_dist, x) ≈ (binomial(nleft, x) * binomial(nright, x))
                        @test all(sufficientstatistics(prod_dist, x) .≈ (x,))
                    end
                end
            end
        end
    end
end
